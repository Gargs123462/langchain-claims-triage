from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
parser = StrOutputParser()

DEADLINES = {"auto": 30, "property": 60, "liability": 45}

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


class ClaimState(TypedDict):
    raw_claim: str
    days_since_incident: int
    cleaned_claim: str
    claim_type: str
    severity: str
    deadline_status: str
    final_decision: str


def cleanup_node(state: ClaimState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the raw claim description into one clean, professional paragraph. Do not add facts that weren't stated."),
        ("human", "{claim_text}")
    ])
    chain = prompt | model | parser
    cleaned = chain.invoke({"claim_text": state["raw_claim"]})
    return {"cleaned_claim": cleaned}


def classify_node(state: ClaimState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a claims classifier. Classify the claim by type and severity.

RULES:
- TYPE must be exactly one of: auto, property, liability
- SEVERITY must be exactly one of: low, medium, high
- If the claim mentions injury to a person, SEVERITY is always high, regardless of dollar amount
- If the claim is ambiguous or doesn't clearly fit one type, choose the closest match and note nothing else
- Do not add any text besides the required format

EXAMPLES:
Claim: "My kitchen pipe burst and flooded the floor, no one was hurt."
Output: TYPE: property | SEVERITY: medium

Claim: "A visitor slipped on my icy driveway and broke their wrist."
Output: TYPE: liability | SEVERITY: high

Claim: "Someone scratched my car door in a parking garage."
Output: TYPE: auto | SEVERITY: low

Respond with exactly this format and nothing else: TYPE: <auto/property/liability> | SEVERITY: <low/medium/high>"""),
        ("human", "{cleaned_claim}")
    ])
    chain = prompt | model | parser
    result = chain.invoke({"cleaned_claim": state["cleaned_claim"]})
    claim_type = result.split("TYPE:")[1].split("|")[0].strip().lower()
    severity = result.split("SEVERITY:")[1].strip().lower()
    return {"claim_type": claim_type, "severity": severity}


def deadline_check_node(state: ClaimState) -> dict:
    results = retriever.invoke(f"{state['claim_type']} policy claim filing deadline window")
    policy_text = results[0].page_content if results else ""

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the claim filing deadline in days from this policy text. Respond with ONLY the number, nothing else."),
        ("human", "{policy_text}")
    ])
    chain = extract_prompt | model | parser
    extracted = chain.invoke({"policy_text": policy_text})

    try:
        limit = int(extracted.strip())
    except ValueError:
        limit = DEADLINES.get(state["claim_type"], 30)

    days = state["days_since_incident"]
    source = results[0].metadata.get("source") if results else "fallback table"

    if days <= limit:
        status = f"within deadline ({days}/{limit} days) - source: {source}"
    else:
        status = f"PAST DEADLINE ({days}/{limit} days) - source: {source}"
    return {"deadline_status": status}


def auto_process_node(state: ClaimState) -> dict:
    return {"final_decision": f"AUTO-APPROVED for processing. Type: {state['claim_type']}, Severity: {state['severity']}, Deadline: {state['deadline_status']}"}


def human_review_node(state: ClaimState) -> dict:
    return {"final_decision": f"FLAGGED FOR HUMAN REVIEW. Type: {state['claim_type']}, Severity: {state['severity']}, Deadline: {state['deadline_status']}"}


def route_decision(state: ClaimState) -> str:
    if "PAST DEADLINE" in state["deadline_status"] or state["severity"] == "high":
        return "human_review"
    return "auto_process"


graph = StateGraph(ClaimState)
graph.add_node("cleanup", cleanup_node)
graph.add_node("classify", classify_node)
graph.add_node("deadline_check", deadline_check_node)
graph.add_node("auto_process", auto_process_node)
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "cleanup")
graph.add_edge("cleanup", "classify")
graph.add_edge("classify", "deadline_check")
graph.add_conditional_edges(
    "deadline_check",
    route_decision,
    {"auto_process": "auto_process", "human_review": "human_review"}
)
graph.add_edge("auto_process", END)
graph.add_edge("human_review", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "raw_claim": "my car got hit in the parking lot some guy backed into me while i was at the grocery store yesterday afternoon bumper is messed up",
		"days_since_incident": 25
    })
    print("CLEANED:", result["cleaned_claim"])
    print("TYPE:", result["claim_type"])
    print("SEVERITY:", result["severity"])
    print("DEADLINE:", result["deadline_status"])
    print("DECISION:", result["final_decision"])