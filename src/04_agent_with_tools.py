from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# Reconnect to the Chroma DB we already built in Phase 2
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def lookup_policy(question: str) -> str:
    """Search the insurance policy documents for relevant coverage information. Use this when you need to know what a policy covers, excludes, or requires."""
    results = retriever.invoke(question)
    return "\n\n".join([f"[{doc.metadata.get('source')}] {doc.page_content}" for doc in results])


@tool
def check_claim_deadline(policy_type: str, days_since_incident: int) -> str:
    """Check if a claim is still within the filing deadline. policy_type must be 'auto', 'property', or 'liability'. days_since_incident is how many days have passed since the incident."""
    deadlines = {"auto": 30, "property": 60, "liability": 45}
    limit = deadlines.get(policy_type.lower())
    if limit is None:
        return f"Unknown policy type: {policy_type}"
    if days_since_incident <= limit:
        return f"Within deadline. {policy_type} claims must be filed within {limit} days; {days_since_incident} days have passed."
    else:
        return f"PAST DEADLINE. {policy_type} claims must be filed within {limit} days; {days_since_incident} days have passed. Flag for manual review."


@tool
def flag_for_human_review(reason: str) -> str:
    """Flag this claim for a human adjuster to review manually. Use this when the claim is past deadline, coverage is unclear, or the situation is unusual."""
    return f"Claim flagged for human review. Reason: {reason}"


tools = [lookup_policy, check_claim_deadline, flag_for_human_review]

agent = create_agent(
    model="gpt-4.1-mini",
    tools=tools,
    system_prompt="You are a claims triage assistant. Use the available tools to research coverage questions, check filing deadlines, and flag claims for human review when needed. Always explain your reasoning."
)

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "A customer's car was hit in a parking lot by an identified driver 25 days ago. Is this covered, and are they still within the filing window?"}
        ]
    })
    print(result["messages"][-1].content)