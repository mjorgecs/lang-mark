from langlib.demo.routes import answer_grader
from langlib.demo.routes import question_rewriter


MAX_REWRITES = 3
DEFAULT_ANSWER = "I don't have enough information to address your question."

def grade_generation(llm, state):
  """
  Determines whether the generation is grounded in the document and answers question.

  Args:
    state (dict): The current graph state

  Returns:
    str: Decision for next node to call
  """

  question = state["question"]
  generation = state["generation"]
  counter = state["counter"]

  ans_grader = answer_grader(llm)

  score = ans_grader.invoke({"question": question, "generation": generation})
  grade = score.binary_score

  if grade == "yes":
    print("---DECISION: GENERATION ADDRESSES QUESTION---")
    return "useful"

  elif counter >= MAX_REWRITES:
    print("---DECISION: CANNOT ADDRESS QUESTION---")
    return "cancel"

  else:
    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
    return "not useful"

# ----------------

def decide_to_generate(state):
  """
  Determines whether to generate an answer, or re-generate a question.

  Args:
    state (dict): The current graph state

  Returns:
    str: Binary decision for next node to call
  """

  print("---ASSESS GRADED DOCUMENTS---")
  filtered_documents = state["documents"]
  counter = state["counter"]


  if filtered_documents:
    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"

  elif counter >= MAX_REWRITES:
    # Maximum number of query transformations reached, cancel the query
    print("---DECISION: CANNOT ADDRESS QUESTION---")
    return "cancel"

  else:
    # We will re-generate a new query
    print(
      "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
    )
    return "transform_query"


def transform_query(llm, state):
  """
  Transform the query to produce a better question.

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): Updates question key with a re-phrased question and increase the counter
  """

  print("---TRANSFORM QUERY---")
  question = state["question"]
  documents = state["documents"]
  counter = state.get("counter", 0)

  quest_re = question_rewriter(llm)

  # Re-write question
  better_question = quest_re.invoke({"question": question})
  return {"documents": documents, "question": better_question, "counter": counter+1}


def cancel_query(state):
  """Replace the generation with a fallback answer after exhausting rewrites."""
  print("---CANCEL: RETURNING DEFAULT ANSWER---")
  return {"generation": DEFAULT_ANSWER}
