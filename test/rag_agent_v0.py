from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"

if not load_dotenv(dotenv_path=env_path):
  raise RuntimeError(f"no .env found at {env_path}")

from typing import List
from functools import partial
from langchain_chroma import Chroma
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from langlib.demo.nodes import retrieve, grade_documents, generate, transform_query
from rag_agent_lib_test import decide_to_generate, grade_generation_v_documents_and_question


DB_PATH = Path(__file__).parent.parent / "db" / "rag_agent"

### SETUP

# Set up the AI embedding model
embeddings = OpenAIEmbeddings(
  model= "text-embedding-3-small"
)

llm = ChatOpenAI(
  model= "gpt-5-mini",
  temperature = 0,
)



### CREATE INDEX

collection_name = "wiki-articles"
DB_PATH.mkdir(parents=True, exist_ok=True)


def load_and_tokenize_urls(urls: List[str]):
  """Load a list of URLs and split it into chunks. Only needed the first time we index."""

  # docs is a list of lists: [[Document], [Document], [Document]]
  docs = [WebBaseLoader(url).load() for url in urls]
  docs_list = [item for sublist in docs for item in sublist]

  # Split
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=0
  )

  return text_splitter.split_documents(docs_list)


urls = [
  "https://simple.wikipedia.org/wiki/Photosynthesis",
  "https://simple.wikipedia.org/wiki/Coffee",
  "https://simple.wikipedia.org/wiki/Volcano"
]


try:

  vectorstore = Chroma(
    collection_name = collection_name,
    embedding_function = embeddings,
    persist_directory = str(DB_PATH),
  )

  if not vectorstore.get(limit=1)["ids"]:
    print("No existing index found, embedding documents")

    docs_split = load_and_tokenize_urls(urls)

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


# --- LLM
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
print("\n======BEGIN SESSION=======\n")

while True:
  user_input = input("🤠 USER: ").strip()

  if user_input.lower() in ("quit", "exit"):
    print("\n======FINNISH SESSION=======")
    break

  if not user_input:
    continue

  try:

    inputs = {
      "question": user_input
    }

    print("\n---⚙️ RAG PROCESS STARTED ⚙️---\n")

    for output in app.stream(inputs):
      for key, value in output.items():
        print(f"Node '{key}':")

    print("---⚙️ RAG PROCESS ENDED⚙️---\n")

    # Final generation
    print(f"🤖 RAG: {value["generation"]}")

  except Exception as e:
    print(f"\nError calling the model: {e}\n")
    continue
