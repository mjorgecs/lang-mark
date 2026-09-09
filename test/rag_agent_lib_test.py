from langlib.demo.routes import answer_grader, hallucination_grader, question_rewriter
from langlib.demo.formatting import format_docs



def decide_to_generate(state):
  """
  Determines whether to generate an answer, or re-generate a question.

  Args:
    state (dict): The current graph state

  Returns:
    str: Binary decision for next node to call
  """

  print("---ASSESS GRADED DOCUMENTS---")
  state["question"]
  filtered_documents = state["documents"]

  if not filtered_documents:
    # All documents have been filtered check_relevance
    # We will re-generate a new query
    print(
      "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
    )
    return "transform_query"
  else:
    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation_v_documents_and_question(llm, state):
  """
  Determines whether the generation is grounded in the document and answers question.

  Args:
    state (dict): The current graph state

  Returns:
    str: Decision for next node to call
  """

  print("---CHECK HALLUCINATIONS---")
  question = state["question"]
  documents = state["documents"]
  generation = state["generation"]

  hallu_grader = hallucination_grader(llm)
  ans_grader = answer_grader(llm)

  score = hallu_grader.invoke(
    {"documents": format_docs(documents), "generation": generation}
  )
  grade = score.binary_score

  # Check hallucination
  if grade == "yes":
    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
    # Check question-answering
    print("---GRADE GENERATION vs QUESTION---")
    score = ans_grader.invoke({"question": question, "generation": generation})
    grade = score.binary_score
    if grade == "yes":
      print("---DECISION: GENERATION ADDRESSES QUESTION---")
      return "useful"
    else:
      print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
      return "not useful"
  else:
    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"


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
