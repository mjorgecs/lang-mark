from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"

if not load_dotenv(dotenv_path=env_path):
  raise RuntimeError(f"no .env found at {env_path}")

from langchain_openai import ChatOpenAI



llm = ChatOpenAI(
  model="gpt-5-mini",
  temperature=3
)


print("\n======BEGIN SESSION=======\n")

while True:
  user_input = input("🤠 USER: ").strip()

  if user_input.lower() in ("quit", "exit"):
    print("\n======FINNISH SESSION=======")
    break
  if not user_input:
    continue

  try:
    response = llm.invoke(user_input)
  except Exception as e:
    print(f"\nError calling the model: {e}\n")
    continue

  print(f"🤖 AI: {response.text}\n")
