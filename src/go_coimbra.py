import os
from dotenv import load_dotenv

load_dotenv()

from pprint import pprint
from functools import partial
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


from edges import decide_to_generate, grade_generation_v_documents_and_question
from nodes import retrieve, grade_documents, transform_query, generate, GraphState


PDF_PATH = Path(__file__).parent.parent / "docs" / "go_coimbra" / "700-maiores-empresas-coimbra-2025-tables.pdf"
COLLECTIONS_PATH = Path(__file__).parent.parent / "collections"

### SETUP

# Set up the AI embedding model
embeddings = GoogleGenerativeAIEmbeddings(
  model= "gemini-embedding-001",
)

llm = ChatGoogleGenerativeAI(
  model= "gemini-3.7-flash",
  temperature = 0,
  max_retries = 2,
)


### CREATE INDEX

collection_name = "wiki-articles"
COLLECTIONS_PATH.mkdir(parents=True, exist_ok=True)


def load_and_split_pdf():
  """Load the PDF and split it into chunks. Only needed the first time we index."""

  if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

  pdf_loader = PyPDFLoader(str(PDF_PATH)) # This loads the PDF

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
  return text_splitter.split_documents(pages)


def load_and_split_urls(urls: List[str]):
  """Load a list of URLs and split it into chunks. Only needed the first time we index."""

  # docs is a list of lists: [[Document], [Document], [Document]]
  docs = [WebBaseLoader(url).load() for url in urls]
  docs_list = [item for sublist in docs for item in sublist]

  # Split
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
  )

  return text_splitter.split_documents(docs_list)



urls = [
  "https://simple.wikipedia.org/wiki/Photosynthesis",
]


# Open the ChromaDB, reusing the one on disk if we already indexed the PDF
try:

  vectorstore = Chroma(
    collection_name = collection_name,
    embedding_function = embeddings,
    persist_directory = str(COLLECTIONS_PATH),
  )

  # An empty collection means we have never indexed the PDF into this directory
  if not vectorstore.get(limit=1)["ids"]:
    print("No existing index found, embedding the PDF (this spends API quota)...")

    docs_split = load_and_split_urls(urls)

    vectorstore.add_documents(docs_split)

    print(f"Created ChromaDB vector store with {len(docs_split)} chunks!")
  else:
    print("Reusing the ChromaDB vector store already on disk, nothing to embed")

except Exception as e:
  print(f"Error setting up ChromaDB: {str(e)}")
  raise


# Create the retriever
retriever = vectorstore.as_retriever(
  search_type="similarity",
  search_kwargs={"k": 2}
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
  "question": "What substances are produced during photosynthesis?"
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
