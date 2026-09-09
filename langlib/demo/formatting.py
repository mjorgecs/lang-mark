from typing import List

from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
  """Flatten documents to plain text for prompt interpolation.

  Prompt templates format via str.format, so passing a list of Document
  objects interpolates their repr - uuid and metadata included, which on
  this corpus costs ~80% more tokens than the text itself.
  """

  return "\n\n".join(d.page_content for d in docs)
