from importlib import import_module

triage = import_module("src.05_langgraph_triage")

graph_image = triage.app.get_graph().draw_mermaid_png()

with open("docs/claims_triage_graph.png", "wb") as f:
    f.write(graph_image)

print("Saved graph diagram to docs/claims_triage_graph.png")