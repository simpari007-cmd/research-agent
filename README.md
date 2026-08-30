# Autonomous Financial Research Agent

Given a ticker and a research question, this agent decides for itself what to investigate, pulls real data via tool calls, evaluates whether it has enough evidence, and either keeps investigating or writes up findings with sources, instead of following a fixed, scripted pipeline.

## Why an agentic loop, not RAG

Most "AI + finance" projects retrieve a fixed set of documents once and summarize them. This agent instead runs a real plan, act, observe, replan loop: at each step, Claude decides which tool(s) to call based on what it's learned so far, not a hardcoded sequence. That decision-making, visible in the reasoning trace, is the actual differentiator.

## Architecture

**Tools** (all real, live data, no mocked responses):
- `get_financial_data` pulls quarterly financial metrics (e.g. gross margin) via Yahoo Finance, computing derived metrics (like margin ratios) from raw line items
- `get_news` pulls recent company news, filtered for relevance to a topic via a single batched Claude call (not per-article, to control cost)
- `get_filing_section` pulls a specific section (e.g. Risk Factors, MD&A) from a company's latest SEC 10-K filing, parsed and cleaned from raw HTML

**Loop control:**
- Hard-capped at 5 tool-call steps before the agent must answer with what it has, and forced to actually synthesize an answer at that point rather than stall
- Handles parallel tool calls correctly (Claude often requests multiple independent tools in a single step, confirmed via live testing that every `tool_use` block requires a matching `tool_result` or the API rejects the whole request)
- Every step logs what was called, why, and what came back, this is the reasoning trace

**A real trust safeguard, found and fixed during testing:**
Early runs showed Claude generating confident-sounding explanations even when no supporting news was found, presenting inference as if it were a data-backed finding. Fixed with a standing system instruction requiring any unconfirmed reasoning to be explicitly labeled `"Inferred (not confirmed by available data):"`, kept clearly separate from tool-grounded findings. An agent that's honest about the limits of its own evidence is a stronger result than one that always sounds equally sure of itself.

## UI

A Streamlit front end (`app.py`) shows the full reasoning trace live as the agent works: each tool call, its input, and its result in an expandable step, plus a tabbed view separating the trace from the final answer. Includes example question chips and a lightweight verdict badge (improving/deteriorating) based on the answer content.

## Example run

Question: *Is NVDA's gross margin improving or deteriorating, and why?*

The agent calls `get_financial_data` for the metric history and `get_news` for context in a single step, then answers:

> NVDA's gross margin is improving significantly. Gross margin rose from 60.5% (Q1 FY2025) to 74.9% (Q1 FY2026).
>
> Inferred (not confirmed by available data): the improvement likely reflects a favorable product mix shift toward high-margin AI accelerators, manufacturing scale efficiencies, and continued pricing power.

A second run, asking about AAPL's risk factors instead, correctly chose `get_filing_section` with `"risk factors"` (a different tool and section than the margin question) with no code changes, demonstrating the loop generalizes across companies and question types rather than being hardcoded to one scenario.

## Tech stack

Python, Anthropic Claude API (tool use / function calling), yfinance, SEC EDGAR (direct API), BeautifulSoup, Streamlit

## Running it

Terminal version:
```
pip3 install anthropic yfinance beautifulsoup4 requests
export ANTHROPIC_API_KEY="your-key-here"
python3 research_agent_loop.py
```

UI version:
```
pip3 install streamlit
export ANTHROPIC_API_KEY="your-key-here"
streamlit run app.py
```

## Status

v1, core loop, all three tools, the trust-labeling safeguard, and a working Streamlit UI are all built and verified against live runs across multiple tickers and question types. Next steps: broader automated testing across more companies, and persistent memory across runs.