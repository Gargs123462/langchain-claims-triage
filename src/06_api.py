from fastapi import FastAPI
from pydantic import BaseModel
from importlib import import_module

triage = import_module("src.05_langgraph_triage")

app = FastAPI(title="Claims Triage API")


class ClaimRequest(BaseModel):
    raw_claim: str
    days_since_incident: int


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/triage-claim")
def triage_claim(request: ClaimRequest):
    result = triage.app.invoke({
        "raw_claim": request.raw_claim,
        "days_since_incident": request.days_since_incident
    })
    return {
        "cleaned_claim": result["cleaned_claim"],
        "claim_type": result["claim_type"],
        "severity": result["severity"],
        "deadline_status": result["deadline_status"],
        "final_decision": result["final_decision"]
    }