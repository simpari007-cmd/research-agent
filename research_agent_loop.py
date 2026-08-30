from anthropic import Anthropic
from tools import get_financial_data, get_news, get_filing_section

client = Anthropic()

MAX_STEPS = 5
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a financial research agent. When analyzing a company, base your findings primarily on the data returned by your tools (financial metrics, news, filings).

If you don't have tool-based evidence for part of your analysis, especially explaining *why* a trend occurred, you may still offer a plausible explanation, but you must clearly label it as inferred, not confirmed. Use a phrase like "Inferred (not confirmed by available data):" before any such reasoning, and keep it clearly separate from findings that are directly backed by tool results.

Never present unconfirmed reasoning with the same confidence or phrasing as data-backed findings."""

tools = [
    {
        "name": "get_financial_data",
        "description": "Get financial metrics (e.g. revenue, gross margin) for a given ticker over a number of recent quarters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker symbol, e.g. NVDA"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metrics to fetch, e.g. ['revenue', 'gross_margin']"
                },
                "periods": {"type": "integer", "description": "Number of recent quarters to fetch, defaults to 8"}
            },
            "required": ["ticker", "metrics"]
        }
    },
    {
        "name": "get_news",
        "description": "Get recent news for a company, optionally filtered by topic (e.g. 'earnings', 'margins').",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker symbol, e.g. NVDA"},
                "topic": {"type": "string", "description": "Optional topic to filter news relevance, e.g. 'margins'"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_filing_section",
        "description": "Get a specific section (e.g. 'risk factors', 'mdna') from a company's latest 10-K filing. Use this for detailed, official management commentary on financial trends, especially margin drivers, that news articles might not cover.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker symbol, e.g. NVDA"},
                "section": {"type": "string", "description": "Which section to retrieve, e.g. 'risk factors' or 'mdna'"}
            },
            "required": ["ticker", "section"]
        }
    }
]

tool_functions = {
    "get_financial_data": get_financial_data,
    "get_news": get_news,
    "get_filing_section": get_filing_section,
}


def execute_tool_calls(response):
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    tool_results = []

    for block in tool_use_blocks:
        print(f"  -> calling {block.name} with input {block.input}")
        func = tool_functions[block.name]
        try:
            result = func(**block.input)
        except Exception as e:
            result = {"error": f"Tool execution failed: {str(e)}"}
        print(f"  <- result: {str(result)[:200]}...")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(result)
        })

    return tool_results


def run_research_loop(ticker, question):
    conversation = [{"role": "user", "content": question}]
    step_count = 0

    print(f"\n=== Research question: {question} ===")

    while step_count < MAX_STEPS:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=conversation
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\nStep {step_count + 1} reasoning: {block.text}")

        if response.stop_reason == "tool_use":
            tool_results = execute_tool_calls(response)
            conversation.append({"role": "assistant", "content": response.content})
            conversation.append({"role": "user", "content": tool_results})
            step_count += 1
        else:
            print(f"\n=== Claude answered after {step_count} step(s) ===")
            return response

    # Loop exited because we hit the cap — force a real, synthesized answer
    print(f"\n=== Hit {MAX_STEPS}-step cap — forcing final answer ===")
    conversation.append({
        "role": "user",
        "content": "You've reached the maximum number of research steps. "
                    "Provide your best final answer now, using only the information "
                    "already gathered above. Do not request further research."
    })
    final_response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=conversation
    )
    return final_response


if __name__ == "__main__":
    ticker = input("Enter a ticker (e.g. NVDA): ").strip().upper()
    question = input("Enter your research question: ").strip()

    result = run_research_loop(ticker, question)

    print("\n=== FINAL ANSWER ===")
    for block in result.content:
        if block.type == "text":
            print(block.text)