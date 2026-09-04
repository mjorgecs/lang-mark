import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
  model= "gemini-3.5-flash",
  temperature = 0.0,
  max_retries = 2,
  google_api_key = api_key,
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