from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
parser = StrOutputParser()

cleanup_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a claims intake assistant. Rewrite the raw claim description into one clean, professional paragraph. Do not add facts that weren't stated."),
    ("human", "{claim_text}")
])

classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a claims triage classifier. Given a claim description, respond with exactly one line in this format: TYPE: <auto/property/liability/other> | SEVERITY: <low/medium/high>. No other text."),
    ("human", "{cleaned_claim}")
])

cleanup_chain = cleanup_prompt | model | parser
classify_chain = classify_prompt | model | parser

full_chain = (
    {"cleaned_claim": cleanup_chain}
    | classify_chain
)

if __name__ == "__main__":
    raw_claim = "my car got hit in the parking lot some guy backed into me while i was at the grocery store yesterday afternoon bumper is messed up"
    
    cleaned = cleanup_chain.invoke({"claim_text": raw_claim})
    print("CLEANED:", cleaned)
    
    classification = classify_chain.invoke({"cleaned_claim": cleaned})
    print("CLASSIFICATION:", classification)