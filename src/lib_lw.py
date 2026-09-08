from routes import answer_grader

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
    generation = state["generation"]

    ans_grader = answer_grader(llm)

    grade = ans_grader.invoke({"question": question, "generation": generation})

    if grade == "yes":
      print("---DECISION: GENERATION ADDRESSES QUESTION---")
      return "useful"
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
  counter = state["re_write_counter"]

  if not filtered_documents and counter < 3:
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
