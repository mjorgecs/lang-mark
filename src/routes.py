from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


### Retrieval Grader-------------------------------------------
# Data model
class GradeDocuments(BaseModel):
  """Binary score for relevance check on retrieved documents."""

  binary_score: str = Field(
    description="Documents are relevant to the question, 'yes' or 'no'"
  )

def retrieval_grader(llm):
  structured_llm_grader = llm.with_structured_output(GradeDocuments)

  # Prompt
  system = """You are a grader assessing relevance of a retrieved document to a user question. \n
      If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
      It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
      Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

  grade_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
  )

  return grade_prompt | structured_llm_grader


### Generate-------------------------------------------
# Prompt
def rag_chain(llm):
  prompt = ChatPromptTemplate.from_messages([
    ("human",
     "You are an assistant for question-answering tasks. Use the following pieces of "
     "retrieved context to answer the question. If you don't know the answer, just say "
     "that you don't know. Use three sentences maximum and keep the answer concise.\n"
     "Question: {question}\nContext: {context}\nAnswer:"),
  ])

  return prompt | llm | StrOutputParser


### Hallucination Grader-------------------------------------------
# Data model
class GradeHallucinations(BaseModel):
  """Binary score for hallucination present in generation answer."""

  binary_score: str = Field(
    description="Answer is grounded in the facts, 'yes' or 'no'"
  )

def hallucination_grader(llm):
  structured_llm_grader = llm.with_structured_output(GradeHallucinations)

  # Prompt
  system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
      Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

  hallucination_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
  )

  return hallucination_prompt | structured_llm_grader


### Answer Grader-------------------------------------------
# Data model
class GradeAnswer(BaseModel):
  """Binary score to assess answer addresses question."""

  binary_score: str = Field(
    description="Answer addresses the question, 'yes' or 'no'"
  )

def answer_grader(llm):
  structured_llm_grader = llm.with_structured_output(GradeAnswer)

  # Prompt
  system = """You are a grader assessing whether an answer addresses / resolves a question \n
      Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

  answer_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
  )

  return answer_prompt | structured_llm_grader


### Question Re-writer-------------------------------------------
def question_rewriter(llm):
  # Prompt
  system = """You a question re-writer that converts an input question to a better version that is optimized \n
      for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

  re_write_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      (
        "human",
        "Here is the initial question: \n\n {question} \n Formulate an improved question.",
      ),
    ]
  )

  return re_write_prompt | llm | StrOutputParser()
