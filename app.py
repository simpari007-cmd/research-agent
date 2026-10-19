import streamlit as st
from anthropic import Anthropic
from tools import get_financial_data, get_news, get_filing_section

client = Anthropic()

MAX_STEPS = 5
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a financial research agent. When analyzing a company, base your findings primarily on the data returned by your tools (financial metrics, news, filings).

If you don't have tool-based evidence for part of your analysis, especially explaining *why* a trend occurred, you may still offer a plausible explanation, but you must clearly label it as inferred, not confirmed. Use a phrase like "Inferred (not confirmed by available data):" before any such reasoning, and keep it clearly separate from findings that are directly backed by tool results.

Never present unconfirmed reasoning with the same confidence or phrasing as data-backed findings.

For any question about growth, trends, acceleration or deceleration, compare year over year (the same quarter in the prior year), not sequentially between adjacent quarters. Sequential comparisons confuse seasonality with trend. Many companies have a structurally strong or weak quarter every year, so a quarter-over-quarter decline is often normal seasonality rather than deterioration. Always state which comparison basis you used. If you only have sequential data and cannot make a year over year comparison, say so explicitly rather than drawing a trend conclusion from adjacent quarters.

This is a multi-turn conversation and the user can ask follow-up questions, so you may reference your earlier findings. Answer the question asked and stop. Do not append an offer to do more research unless the user's question is genuinely ambiguous.

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

TOOL_ICONS = {
    "get_financial_data": "📈",
    "get_news": "📰",
    "get_filing_section": "📄",
}

MAX_STEPS_NUDGE = (
    "You've reached the maximum number of research steps. Provide your best "
    "final answer now, using only the information already gathered above. "
    "Do not request further research."
)


def run_research_loop(conversation, trace):
    """Run the plan, act, observe loop against the shared conversation.

    Appends every assistant turn to `conversation` so follow-up questions keep
    the full context, and records each step into `trace` as plain data so the
    UI can re-render it on later Streamlit reruns.

    Returns the final answer text.
    """
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
                trace.append({
                    "kind": "reasoning",
                    "step": step_count + 1,
                    "text": block.text,
                })

        # Record the assistant turn before anything else, so the conversation
        # stays valid whether or not tools were called.
        conversation.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "\n".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            try:
                result = tool_functions[block.name](**block.input)
            except Exception as e:
                result = {"error": f"Tool execution failed: {str(e)}"}

            trace.append({
                "kind": "tool",
                "step": step_count + 1,
                "name": block.name,
                "input": dict(block.input),
                "result": str(result)[:1500],
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result)
            })

        conversation.append({"role": "user", "content": tool_results})
        step_count += 1

    # Out of steps. Fold the nudge into the existing user turn rather than
    # appending a second consecutive user message, which the Messages API
    # rejects because roles must alternate.
    last = conversation[-1]
    if last["role"] == "user" and isinstance(last["content"], list):
        last["content"].append({"type": "text", "text": MAX_STEPS_NUDGE})
    else:
        conversation.append({"role": "user", "content": MAX_STEPS_NUDGE})

    final = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=conversation
    )
    conversation.append({"role": "assistant", "content": final.content})
    return "\n".join(b.text for b in final.content if b.type == "text")


def verdict_badge(answer_text):
    """Cosmetic only. Returns a badge string or None."""
    lowered = answer_text.lower()
    if "deteriorat" in lowered:
        return "🔴 DETERIORATING"
    if "improving" in lowered:
        return "🟢 IMPROVING"
    return None


def render_turn(turn, index):
    """Re-render one completed question from stored data."""
    st.markdown(f"**Q{index + 1}. {turn['question']}**")

    trace_tab, answer_tab = st.tabs(["🔍 Reasoning trace", "✅ Answer"])

    with trace_tab:
        if not turn["trace"]:
            st.caption("No tool calls were needed for this question.")
        for event in turn["trace"]:
            if event["kind"] == "reasoning":
                with st.expander(f"💭 Step {event['step']}: reasoning"):
                    st.markdown(escape_dollars(event["text"]))
            else:
                icon = TOOL_ICONS.get(event["name"], "🔧")
                with st.expander(f"{icon} Step {event['step']}: called `{event['name']}`"):
                    inputs = ", ".join(f"**{k}**: {v}" for k, v in event["input"].items())
                    st.markdown(f"Input → {inputs}")
                    st.code(event["result"], language="python")

    with answer_tab:
        badge = verdict_badge(turn["answer"])
        if badge:
            st.markdown(f"### {badge}")
        with st.container(border=True):
            st.markdown(escape_dollars(turn["answer"]))



st.set_page_config(page_title="Financial Research Agent", page_icon="📊", layout="centered")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; max-width: 780px; }
    .agent-header { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.1rem; }
    .agent-subtitle { color: #666; font-size: 0.95rem; margin-bottom: 1.4rem; }
    .stButton button { border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="agent-header">📊 Autonomous Financial Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="agent-subtitle">Give it a ticker and a question, it decides for itself what to '
    'investigate, and shows its reasoning at every step. Ask follow-ups and it keeps the context.</div>',
    unsafe_allow_html=True
)


if "history" not in st.session_state:
    st.session_state.history = []       # completed turns, for re-rendering
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # the live API message list
if "ticker" not in st.session_state:
    st.session_state.ticker = None
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

pending_question = None
pending_ticker = None

if not st.session_state.history:
    # First run: the ticker and question form.
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            ticker_input = st.text_input("Ticker", value="NVDA", placeholder="e.g. NVDA")
        with col2:
            question_input = st.text_input("Research question", key="question_text")

        st.caption("Try one of these:")
        chip_cols = st.columns(len(EXAMPLES))
        for col, (label, text) in zip(chip_cols, EXAMPLES):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state.pending_question = text
                    st.rerun()

        if st.button("🚀 Run research", type="primary", use_container_width=True):
            if not ticker_input.strip() or not question_input.strip():
                st.error("Please enter both a ticker and a question.")
            else:
                pending_ticker = ticker_input.strip().upper()
                pending_question = question_input.strip()
else:
    # Follow-up mode.
    header_col, reset_col = st.columns([3, 1])
    with header_col:
        st.caption(f"Researching **{st.session_state.ticker}**")
    with reset_col:
        if st.button("↺ New company", use_container_width=True):
            st.session_state.history = []
            st.session_state.conversation = []
            st.session_state.ticker = None
            st.rerun()

    st.divider()
    for i, turn in enumerate(st.session_state.history):
        render_turn(turn, i)
        st.divider()

    follow_up = st.chat_input(f"Ask a follow-up about {st.session_state.ticker}")
    if follow_up and follow_up.strip():
        pending_ticker = st.session_state.ticker
        pending_question = follow_up.strip()



if pending_question:
    first_turn = not st.session_state.history
    st.session_state.ticker = pending_ticker

    if first_turn:
        prompt = f"Regarding {pending_ticker}: {pending_question}"
    else:
        prompt = pending_question

    st.session_state.conversation.append({"role": "user", "content": prompt})

    trace = []
    with st.spinner(f"🔎 Researching {pending_ticker}..."):
        try:
            answer = run_research_loop(st.session_state.conversation, trace)
        except Exception as e:
            # Roll back the user turn so a failed call does not corrupt the
            # conversation for the next question.
            st.session_state.conversation.pop()
            st.error(f"Research failed: {e}")
            st.stop()

    st.session_state.history.append({
        "question": pending_question,
        "trace": trace,
        "answer": answer,
    })
    st.rerun()
