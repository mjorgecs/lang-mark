import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from typing import List
from typing_extensions import TypedDict


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





### LLMs-------------------------------------------
# Data model
class GraphState(TypedDict):
  """
  Represents the state of our graph.

  Attributes:
    question: question
    generation: LLM generation
    documents: list of documents
  """

  question: str
  generation: str
  documents: List[str]


# Nodes-------------------------------------------
def retrieve(state):
  """
  Retrieve documents

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): New key added to state, documents, that contains retrieved documents
  """

  print("--RETRIEVE--")
  question = state["question"]

  # Retrieval
  documents = retriever.invoke(question)
  return {"documents": documents, "question": question}


def grade_documents(state):
  """
  Determines wether the retrieved documents are relevant to the question.

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): Updates documents key with only filtered relevant documents
  """

  print("---CHECK DUCUMENT RELEVANCE TO QUESTION---")
  question = state["question"]
  documents = state["documents"]

  # Score each document
  filtered_docs = []
  for d in documents:
    score = retrive
