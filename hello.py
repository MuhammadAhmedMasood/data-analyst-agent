from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads .env into environment variables

llm = ChatOpenAI(model="gpt-5.4-mini")

response = llm.invoke("In one sentence, what is DuckDB?")
print(response.content)
print(response.usage_metadata)