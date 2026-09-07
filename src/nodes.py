from typing import List
from typing_extensions import TypedDict
from routes import retriever, rag_chain, retrieval_grader, question_rewriter


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


def retrieve(retriever, state):
  """
  Retrieve documents

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): New key added to state, documents, that contains retrieved documents
  """

  print("--RETRIEVE--")
  question = state["question"]

  documents = retriever.invoke(question)

  return {"documents": documents, "question": question}


def grade_documents(llm, state):
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

  ret_grader = retrieval_grader(llm)

  # Score each document
  filtered_docs = []
  for d in documents:
    score = ret_grader.invoke(
      {"question": question, "document": d.page_content}
    )
    grade = score.binary_score
    if grade == "yes":
      print("---GRADE: DOCUMENT RELEVANT---")
      filtered_docs.append(d)
    else:
      print("---GRADE: DOCUMENT NOT RELEVANT---")
      continue
  return {"documents": filtered_docs, "question": question}


def transform_query(llm, state):
  """
  Transform the query to produce a better question.

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): Updates question key with a re-phrased question
  """

  print("---TRANSFORM QUERY---")
  question = state["question"]
  documents = state["documents"]

  quest_re = question_rewriter(llm)

  # Re-write question
  better_question = quest_re.invoke({"question": question})
  return {"documents": documents, "question": better_question}


def generate(llm, state):
  """
  Generate answer

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): New key added to state, generation, that contains LLM generation
  """
  print("---GENERATE---")
  question = state["question"]
  documents = state["documents"]

  chain = rag_chain(llm)

  # RAG generation
  generation = chain.invoke({"context": documents, "question": question})
  return {"documents": documents, "question": question, "generation": generation}
