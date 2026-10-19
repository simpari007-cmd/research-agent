import streamlit as st
from anthropic import Anthropic
from tools import get_financial_data, get_news, get_filing_section

client = Anthropic()

MAX_STEPS = 5
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a financial research agent. When analyzing a company, base your findings primarily on the data returned by your tools (financial metrics, news, filings).

If you don't have tool-based evidence for part of your analysis, especially explaining *why* a trend occurred, you may still offer a plausible explanation, but you must clearly label it as inferred, not confirmed. Use a phrase like "Inferred (not confirmed by available data):" before any such reasoning, and keep it clearly separate from findings that are directly backed by tool results.

Never present unconfirmed reasoning with the same confidence or phrasing as data-backed findings.

For any question about growth, trends, acceleration or deceleration, compare
year over year (the same quarter in the prior year), not sequentially between
adjacent quarters. Sequential comparisons confuse seasonality with trend. Many
companies have a structurally strong or weak quarter every year, so a
quarter-over-quarter decline is often normal seasonality rather than
deterioration. Always state which comparison basis you used. If you only have
sequential data and cannot make a year over year comparison, say so explicitly
rather than drawing a trend conclusion from adjacent quarters.
 
This runs in a single-turn interface. The user cannot reply to you. Do not ask
follow-up questions and do not offer to continue the analysis. End with your
findings.

Do not use em dashes anywhere in your response. Use commas, periods, or parentheses instead."""


def escape_dollars(text):
    """Streamlit's markdown renderer treats $ as LaTeX math delimiters,
    which mangles dollar amounts like $4.5B. Escape them so they render as plain text."""
    return text.replace("$", "\\$")

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


def run_research_loop_ui(ticker, question, trace_container):
    """Same loop logic as research_agent_loop.py, but writes each step
    to the Streamlit UI live instead of printing to a terminal."""
    full_question = f"Regarding {ticker}: {question}"
    conversation = [{"role": "user", "content": full_question}]
    step_count = 0

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
                with trace_container.expander(f"💭 Step {step_count + 1}: reasoning", expanded=True):
                    st.markdown(escape_dollars(block.text))

        if response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []

            for block in tool_use_blocks:
                icon = TOOL_ICONS.get(block.name, "🔧")
                with trace_container.expander(f"{icon} Step {step_count + 1}: called `{block.name}`", expanded=True):
                    input_str = ", ".join(f"**{k}**: {v}" for k, v in block.input.items())
                    st.markdown(f"Input → {input_str}")
                    func = tool_functions[block.name]
                    try:
                        result = func(**block.input)
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {str(e)}"}
                    with st.popover("View raw result"):
                        st.code(str(result)[:1500], language="python")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

            conversation.append({"role": "assistant", "content": response.content})
            conversation.append({"role": "user", "content": tool_results})
            step_count += 1
        else:
            return response

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


# ── UI ───────────────────────────────────────────────────────────────

# ── UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Financial Research Agent", page_icon="📊", layout="centered")

TOOL_ICONS = {
    "get_financial_data": "📈",
    "get_news": "📰",
    "get_filing_section": "📄",
}

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; max-width: 780px; }
    .agent-header { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.1rem; }
    .agent-subtitle { color: #666; font-size: 0.95rem; margin-bottom: 1.4rem; }
    .stButton button { border-radius: 8px; font-weight: 600; }
    div[data-testid="stPopover"] button { font-size: 0.8rem; padding: 0.2rem 0.6rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="agent-header">📊 Autonomous Financial Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="agent-subtitle">Give it a ticker and a question, it decides for itself what to '
    'investigate, and shows its reasoning at every step.</div>',
    unsafe_allow_html=True
)

if "question_text" not in st.session_state:
    st.session_state.question_text = "Is gross margin improving or deteriorating, and why?"

# Apply any pending example-chip click BEFORE the widget below is created
if "pending_question" in st.session_state:
    st.session_state.question_text = st.session_state.pop("pending_question")

EXAMPLES = [
    ("📈 Margin trend", "Is gross margin improving or deteriorating, and why?"),
    ("⚠️ Risk factors", "What are the main risk factors right now?"),
    ("💰 Revenue growth", "Is revenue growth accelerating or slowing down?"),
]

with st.container(border=True):
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Ticker", value="NVDA", placeholder="e.g. NVDA")
    with col2:
        question = st.text_input("Research question", key="question_text")

    st.caption("Try one of these:")
    chip_cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(chip_cols, EXAMPLES):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.pending_question = text
                st.rerun()

    run_clicked = st.button("🚀 Run research", type="primary", use_container_width=True)

if run_clicked:
    if not ticker.strip() or not question.strip():
        st.error("Please enter both a ticker and a question.")
    else:
        st.divider()

        status = st.empty()
        status.info(f"🔎 Researching **{ticker.strip().upper()}**...")

        trace_tab, answer_tab = st.tabs(["🔍 Reasoning trace", "✅ Final answer"])

        with trace_tab:
            trace_container = st.container()
            result = run_research_loop_ui(ticker.strip().upper(), question.strip(), trace_container)

        status.empty()

        # crude verdict badge based on keywords in the answer, purely cosmetic
        answer_text = "\n".join(b.text for b in result.content if b.type == "text")
        lowered = answer_text.lower()
        if "improving" in lowered and "deteriorating" not in lowered.split("improving")[0][-50:]:
            badge = "🟢 IMPROVING"
        elif "deteriorat" in lowered:
            badge = "🔴 DETERIORATING"
        else:
            badge = None

        with answer_tab:
            if badge:
                st.markdown(f"### {badge}")
            with st.container(border=True):
                st.markdown(escape_dollars(answer_text))

        st.toast("Research complete", icon="✅")
