from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a claims intake assistant. Rewrite the raw claim description into one clean, professional paragraph. Do not add facts that weren't stated."),
    ("human", "{claim_text}")
])

parser = StrOutputParser()

chain = prompt | model | parser

if __name__ == "__main__":
    raw_claim = "my car got hit in the parking lot some guy backed into me while i was at the grocery store yesterday afternoon bumper is messed up"
    result = chain.invoke({"claim_text": raw_claim})
    print(result)