from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated, List
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

from tools import TOOLS  # ✅ Import the full tools list (includes read_image_text)

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT = 5000


# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# -------------------------------------------------
# GEMINI LLM
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9 / 60,  # 9 requests per minute
    check_every_n_seconds=1,
    max_bucket_size=9,
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter,
).bind_tools(TOOLS)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load the quiz page from the given URL.
2. Extract ALL instructions, required parameters, submission rules, and the submit endpoint.
3. Solve the task exactly as required.
4. Submit the answer ONLY to the endpoint specified on the current page (never make up URLs).
5. Read the server response and:
   - If it contains a new quiz URL → fetch it immediately and continue.
   - If it has "correct": true and "url": null → the quiz chain is finished; ONLY THEN return "END".
   - If it has "correct": false or a message like "Incorrect answer" or "you can continue" → DO NOT return END. Fix your logic and try again on the SAME quiz URL.

STRICT RULES — FOLLOW EXACTLY:

GENERAL RULES:
- NEVER stop early. Continue solving tasks until a server response has "correct": true AND "url": null.
- NEVER hallucinate URLs, endpoints, fields, values, or JSON structure.
- NEVER shorten or modify URLs. Always submit the full URL.
- NEVER re-submit unless the server explicitly allows or it's within the 3-minute limit.
- ALWAYS inspect the server response JSON (correct, message/reason, url, delay) before deciding what to do next.
- ALWAYS use the tools provided to fetch, scrape, download, render HTML, or send requests.

TOOLS BEHAVIOR:
- Use get_rendered_html(url) to load and render quiz pages.
- Use download_file(url) to download CSV, Excel, PDF, audio, or image files.
- Use run_code(code) for any non-trivial data processing or analysis.
- Use post_request(url, json) to submit answers to quiz backends.
- Use add_dependencies(packages) if a required Python library is missing.
- Use read_image_text(image_url) when the question refers to text or a "secret code" inside an image on the page.
  First convert any relative image path into a full absolute URL, then call this tool.
  Never guess the code from the image – always read it using this tool.

AUDIO BEHAVIOR (VERY IMPORTANT):
- When the page mentions an audio file (e.g. .wav, .mp3) or asks you to "listen" to something:
  1. Find the exact audio URL from the HTML.
  2. If it's a relative path, convert it to a full absolute URL based on the page URL.
  3. Call download_file(audio_url) to download the audio file.
  4. Then call run_code(...) to:
     - Load the downloaded file from disk.
     - Use a Python speech-to-text library (e.g. speech_recognition) to transcribe.
     - If the library is missing, first call add_dependencies(["speechrecognition", "pydub"]) or similar.
  5. Extract the exact phrase / code required by the question and use that as the answer.
- NEVER guess what is in the audio. ALWAYS download and process it using code.

SERVER RESPONSE LOGIC:
- After every post_request, carefully examine the JSON:
  - If "correct": true and "url" is a non-null URL → navigate to that URL and continue solving.
  - If "correct": true and "url" is null → this quiz chain is fully complete. In your next assistant message, respond with exactly: END
  - If "correct": false OR the message contains phrases like "Incorrect answer", "Key mismatch", "Wrong sum", or "you can continue":
      - DO NOT return END.
      - Re-analyse the current page and your previous logic or code.
      - Correct your mistake and submit a new answer for the SAME quiz URL, as long as the time limit is not exceeded.

TIME LIMIT RULES:
- Each task has a hard 3-minute limit.
- The server response may include a "delay" field indicating elapsed time.
- If your answer is wrong, you may retry, but be mindful of the overall time.

STOPPING CONDITION (VERY IMPORTANT):
- Only return "END" when:
  - A server response has "correct": true AND
  - "url": null (no further quiz URL is provided).
- DO NOT return END in any of these cases:
  - "correct": false
  - The message says "Incorrect answer, but you can continue"
  - You are unsure if the quiz chain is finished.

ADDITIONAL INFORMATION YOU MUST INCLUDE WHEN REQUIRED:
- Email: {EMAIL}
- Secret: {SECRET}

YOUR JOB:
- Follow pages exactly.
- Extract data reliably.
- Never guess.
- Use the correct tool when encountering images or data files.
- Submit correct answers.
- Keep working until a response with "correct": true and "url": null is received.
- Then respond with: END
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm_with_prompt = prompt | llm


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}


# -------------------------------------------------
# ROUTING
# -------------------------------------------------
def route(state: AgentState):
    last = state["messages"][-1]

    # Handle tool calls (both LC objects and dicts)
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")

    if tool_calls:
        return "tools"

    # Get content robustly
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")

    # Check for END signal
    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END

    return "agent"


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)

app = graph.compile()


# -------------------------------------------------
# TEST ENTRYPOINT
# -------------------------------------------------
def run_agent(url: str):
    # Run the state machine
    result = app.invoke(
        {"messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    print("Tasks completed succesfully")

    # --- OPTIONAL SAFETY CHECK (best-effort) ---
    # Try to detect the last server JSON in the messages
    import json

    server_json = None

    for msg in reversed(result["messages"]):
        content = None
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content")

        if isinstance(content, str):
            try:
                data = json.loads(content)
                if isinstance(data, dict) and ("correct" in data or "message" in data):
                    server_json = data
                    break
            except Exception:
                continue

    if server_json is None:
        # Not fatal, but useful to know
        print("⚠️  WARNING: Could NOT find a parsed server JSON in messages.")
        return

    if not server_json.get("correct") or server_json.get("url") is not None:
        print("⚠️  WARNING: Final server response suggests the quiz chain is NOT fully finished.")
        print("Last server response:", server_json)
    else:
        print("🎉 QUIZ COMPLETED SUCCESSFULLY & VERIFIED (correct=True, url=None)!")
