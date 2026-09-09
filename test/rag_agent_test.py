from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"

if not load_dotenv(dotenv_path=env_path):
  raise RuntimeError(f"no .env found at {env_path}")

from typing import List
from pprint import pprint
from functools import partial
from langchain_chroma import Chroma
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from langlib.demo.nodes import retrieve, grade_documents, generate
from test.rag_agent_lib_test import decide_to_generate, grade_generation_v_documents_and_question, transform_query


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


try:

  vectorstore = Chroma(
    collection_name = collection_name,
    embedding_function = embeddings,
    persist_directory = str(DB_PATH),
  )

  if not vectorstore.get(limit=1)["ids"]:
    print("No existing index found, embedding the documents")

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

# --- LLM

# Data model
class GraphState(TypedDict):
  """
  Represents the state of our graph.

  Attributes:
    question: question
    generation: LLM generation
    documents: list of documents
    counter: number of query transformations performed
  """

  question: str
  generation: str
  documents: List[str]
  counter: int


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

workflow.add_conditional_edges(
  "generate",
  partial(grade_generation_v_documents_and_question, llm),
  {
    "useful": END,
    "not useful": "transform_query",
  },
)

workflow.add_edge("transform_query", "retrieve")

# Compile
app = workflow.compile()


# Run
inputs = {
  "question": "What substances are produced during photosynthesis?",
  "counter": 0
}

for output in app.stream(inputs):
  for key, value in output.items():
    # Node
    pprint(f"Node '{key}':")
  pprint("\n---\n")

# Final generation
pprint(value["generation"])
