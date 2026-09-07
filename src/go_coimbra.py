import os
from pprint import pprint
from functools import partial
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


from edges import decide_to_generate, grade_generation_v_documents_and_question
from nodes import retrieve, grade_documents, transform_query, generate, GraphState


PDF_PATH = "../docs/do_coimbra/700-maiores-empresas-coimbra-2025-extended.pdf"
COLLECTIONS_PATH = "../collections"

### SETUP

# Load .env info
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')


# Set up the AI embedding model
embeddings = GoogleGenerativeAIEmbeddings(
  model= "gemini-3.5-flash",
  temperature = 0,
  max_retries = 2,
  google_api_key = api_key,
)

llm = ChatGoogleGenerativeAI(
  model= "gemini-3.5-flash",
  temperature = 0,
  max_retries = 2,
  google_api_key = api_key,
)


### CREATE INDEX

# Load the PDF to index into the ChromaDB
if not os.path.exists(PDF_PATH):
  raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

pdf_loader = PyPDFLoader(PDF_PATH) # This loads the PDF


# Checks if the PDF is there
try:
  pages = pdf_loader.load()
  print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
  print(f"Error loading PDF: {e}")
  raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200
)


# Split the pages into chunks
pages_split = text_splitter.split_documents(pages)


# Create the ChromaDB
try:

  collection_name = "700-extended"
  if not os.path.exists(COLLECTIONS_PATH):
    raise FileNotFoundError(F"Directory not found: {COLLECTIONS_PATH}")

  # Here, we actually create the chroma database using our embeddigns model
  vectorstore = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=COLLECTIONS_PATH,
    collection_name=collection_name
  )
  print(f"Created ChromaDB vector store!")

except Exception as e:
  print(f"Error setting up ChromaDB: {str(e)}")
  raise


# Create the retriever
retriever = vectorstore.as_retriever(
  search_type="similarity", # 'similarity' (default), 'mmr', or 'similarity_score_threshold'
  search_kwargs={"k": 5} # K is the amount of chunks to return
)


# INITIALIZE NODES AND EDGES

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", partial(retrieve, retriever))
workflow.add_node("generate", partial(generate, llm))
workflow.add_node("transform_query", partial(transform_query, llm))
workflow.add_node("grade_documents", partial(grade_documents, llm))

# Build graph
workflow.add_edge(START, "retrieve")

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
  "grade_documents",
  decide_to_generate,
  {
    "transform_query": "transform_query",
    "generate": "generate",
  },
)
workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
  "generate",
  partial(grade_generation_v_documents_and_question, llm),
  {
    "not supported": "generate",
    "useful": END,
    "not useful": "transform_query",
  },
)


# Compile
app = workflow.compile()


# Run
inputs = {
  "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
}

for output in app.stream(inputs):
  for key, value in output.items():
    # Node
    pprint(f"Node '{key}':")
    # Optional: print full state at each node
    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
  pprint("\n---\n")

# Final generation
pprint(value["generation"])
