"""
╔══════════════════════════════════════════════════════════════════╗
║              SYSTEM PROMPT & TOOL DEFINITIONS                    ║
║                                                                  ║
║  This file defines WHO the agent is and WHAT it can do.          ║
║  It's the most important file in any agentic system.             ║
╚══════════════════════════════════════════════════════════════════╝

=== CONCEPT: PROMPT ENGINEERING FOR AGENTS ===

A system prompt for an agent is NOT the same as a chatbot prompt.
A chatbot prompt says: "You are a helpful assistant."
An agent prompt must define:

  1. ROLE    — What the agent IS (researcher, coder, analyst)
  2. TOOLS   — What the agent CAN DO (search, read, calculate)
  3. RULES   — What the agent MUST follow (always cite sources)
  4. STOP    — When the agent should FINISH (enough info gathered)

Without clear rules, agents loop forever, hallucinate, or ignore tools.

=== CONCEPT: TOOL DEFINITIONS (FUNCTION CALLING) ===

"Tool use" means the LLM doesn't just generate text — it can REQUEST
that your code execute a function. Here's how it works:

  1. You define tools as JSON schemas (name, description, parameters)
  2. You send these schemas alongside the conversation to the LLM
  3. The LLM can respond with: "I want to call search(query='...')"
  4. Your code actually calls the search API
  5. You send the result back to the LLM
  6. The LLM decides what to do next

The LLM NEVER executes code. It only REQUESTS a tool call. Your Python
code is the one doing the actual work (HTTP requests, file I/O, etc.)

This is the bridge between "AI that talks" and "AI that acts."
"""

# ─── SYSTEM PROMPT ───────────────────────────────────────────────
# This tells the LLM who it is and how it should behave.
# Notice how specific the rules are — vague prompts = vague behavior.

SYSTEM_PROMPT = """You are a research agent. Your job is to answer the user's question accurately using real information from the web.

You have the following tools:
- search(query: str) — searches the web, returns a list of results (title, url, snippet)
- read(url: str) — reads the full text content of a webpage
- finish(answer: str, sources: list[str]) — call this when you are confident in your answer

Rules:
1. ALWAYS search before trying to read any page
2. Read at LEAST 2 different sources before calling finish
3. If a page is irrelevant after reading it, search with a different query
4. NEVER make up information — only use what the tools return to you
5. When you call finish(), your answer must reference specific facts from the sources you read
6. Keep your answer well-structured with clear sections
7. If you cannot find reliable information after multiple attempts, call finish() and honestly state what you could and could not find

You should think step-by-step about what you need to find, then use your tools systematically."""


# ─── TOOL DEFINITIONS ───────────────────────────────────────────
# These are JSON schemas that tell the LLM what tools exist.
#
# Each tool has:
#   - name:         what the LLM calls it (e.g., "search")
#   - description:  helps the LLM decide WHEN to use it
#   - input_schema: defines the parameters (like a function signature)
#
# The LLM uses these descriptions to decide:
#   "Should I search? What query should I use?"
#   "Should I read this URL or a different one?"
#   "Do I have enough info to finish?"
#
# Better descriptions → better tool usage decisions.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web for information. Returns a list of results, "
                "each with a title, URL, and snippet. Use this to find relevant "
                "pages before reading them. Use specific, targeted search queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — be specific and targeted"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read the full text content of a webpage. Use this after searching "
                "to get detailed information from a specific page. Only read pages "
                "that seem relevant based on their search snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the page to read"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this when you have enough information to answer the user's "
                "question confidently. You MUST have read at least 2 sources. "
                "Your answer should be comprehensive and cite the sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete, well-structured answer"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs you used to form your answer"
                    }
                },
                "required": ["answer", "sources"]
            }
        }
    }
]
