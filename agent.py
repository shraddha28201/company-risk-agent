from google import genai
from google.genai import types
import os
from tools import get_company, risk_score, search_policy

MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

SYSTEM = """You are a company risk assistant for a credit analyst.
Rules:
- Use the tools to get facts. Never invent numbers, company data or policy rules.
- For a question about a company, use lookup_company and get_risk_score.
- For questions about what to do next or what a score means, use find_policy.
- If the user does not give a company number, ask for it.
- If a tool returns an error or no policy, say so plainly.
- The risk score is a ranking score, not a probability. Do not say a company will default.
- Never approve or reject a company. Say that a human analyst makes the final decision.
- If missing_fields is 5 or more, say that the score is less reliable and that updated financial statements are needed.
- Never show internal field names such as missing_fields. Use plain words, for example "6 financial fields are missing".
- Keep answers short and simple."""

def get_key():
    if os.path.exists("key.txt"):
        return open("key.txt").read().strip()
    return os.environ.get("GEMINI_API_KEY", "")


client = genai.Client(api_key=get_key())
# wrappers so that Gemini sees clear names, types and descriptions
def lookup_company(company_num: int) -> dict:
    """Get the key financial figures of a company, using its company number."""
    return get_company(company_num)


def get_risk_score(company_num: int) -> dict:
    """Run the trained model and return the risk score, flag and risk level for a company number."""
    return risk_score(company_num)


def find_policy(question: str) -> list:
    """Search the risk policy rules for text that answers the question."""
    return search_policy(question)


def run_agent(question):
    for name in MODELS:
        try:
            resp = client.models.generate_content(
                model=name,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    tools=[lookup_company, get_risk_score, find_policy],
                ),
            )
            steps = []
            for item in resp.automatic_function_calling_history or []:
                for part in item.parts or []:
                    if getattr(part, "function_call", None):
                        steps.append(f"{part.function_call.name}({dict(part.function_call.args)})")
            return resp.text, steps
        except Exception as e:
            print(f"{name} failed: {str(e)[:70]}")
    return "Google is busy right now. Please try again in a few minutes.", []


if __name__ == "__main__":
    print("Company Risk Assistant. Type 'quit' to stop.")
    while True:
        q = input("\nYou: ")
        if q.strip().lower() == "quit":
            break
        answer, steps = run_agent(q)
        print("\nSteps taken:", steps if steps else "none")
        print("\nAssistant:", answer)