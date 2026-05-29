"""
╔══════════════════════════════════════════════════════════════════╗
║                    LLM API WRAPPER                               ║
║                                                                  ║
║  This file handles communication with the LLM via OpenRouter.    ║
║  It's the "brain" connector — everything else is plumbing.       ║
╚══════════════════════════════════════════════════════════════════╝

=== CONCEPT: WHY OPENROUTER? ===

OpenRouter is a unified API gateway for 100+ LLMs. Instead of:
  - Signing up for OpenAI, Anthropic, Mistral separately
  - Managing 3+ different API keys
  - Learning 3+ different API formats

You get ONE key, ONE API format, and access to every model.

The API format follows the OpenAI Chat Completions spec:
  POST https://openrouter.ai/api/v1/chat/completions
  Body: { model, messages, tools }

This is the industry standard — if you learn this format,
you can switch to any provider (OpenAI, Anthropic, etc.) by
just changing the base URL.

=== CONCEPT: WHY httpx INSTEAD OF requests? ===

`requests` is synchronous — when you make an HTTP call, your
entire Python program STOPS and waits. This is fine for simple
scripts, but terrible for agents because:

  1. Agents make MANY HTTP calls (search, read, LLM, read, LLM...)
  2. In Project 5, agents run in PARALLEL — `requests` can't do that
  3. `httpx` supports both sync AND async, so you learn it once

`httpx` with async:
  async with httpx.AsyncClient() as client:
      response = await client.post(...)  # doesn't block!

The `await` keyword means "start this HTTP call, but let other
code run while we wait for the response." This is the async/await
pattern you'll use in every project.
"""

import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ───────────────────────────────────────────────
# OpenRouter endpoint — same format as OpenAI's API
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model selection — you can change this to any model on OpenRouter
# Examples:
#   "mistralai/mistral-7b-instruct"      — cheap, fast, good at tool use
#   "anthropic/claude-3-haiku"            — fast, great at structured output
#   "anthropic/claude-3.5-sonnet"         — best quality, more expensive
#   "google/gemini-2.0-flash-001"         — fast, good balance
#   "meta-llama/llama-3-8b-instruct"      — open source, cheap
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


async def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1
) -> dict:
    """
    Send a conversation + tools to the LLM and get a response.

    === HOW THE CONVERSATION FORMAT WORKS ===

    `messages` is a list of dicts, each with a "role" and "content":

      [
        {"role": "system",    "content": "You are a researcher..."},
        {"role": "user",      "content": "What caused the 2008 crisis?"},
        {"role": "assistant", "content": "I'll search for that..."},
        {"role": "tool",      "content": "search results here..."},
      ]

    The roles are:
      - "system"    → instructions the model always follows (set once)
      - "user"      → what the human asked (or tool results we're feeding back)
      - "assistant" → what the model said previously
      - "tool"      → result of a tool call the model requested

    We send the ENTIRE conversation history every time. The LLM is
    stateless — it doesn't remember previous calls. The history IS
    its memory.

    === WHAT TOOL_CHOICE="auto" MEANS ===

    When we pass `tool_choice: "auto"`, we're telling the LLM:
      "You can call a tool OR just respond with text — your choice."

    Other options:
      "none"     → never use tools (just generate text)
      "required" → MUST use a tool (no plain text response)
      {"type": "function", "function": {"name": "search"}} → MUST use THIS specific tool

    Parameters
    ----------
    messages : list[dict]
        The full conversation history
    tools : list[dict] | None
        Tool definitions the LLM can choose from
    model : str
        Which LLM model to use
    temperature : float
        0.0 = deterministic (same input → same output)
        1.0 = creative (more random)
        For agents, low temperature (0.1) is better — we want
        consistent, reliable decisions, not creative writing.

    Returns
    -------
    dict with keys:
        - "content": the text response (if any)
        - "tool_calls": list of tool calls the LLM wants to make (if any)
        - "finish_reason": why the LLM stopped ("stop" or "tool_calls")
    """
    if not API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not found!\n"
            "1. Copy .env.example to .env\n"
            "2. Add your key from https://openrouter.ai/keys"
        )

    # Build the request payload
    # This follows the OpenAI Chat Completions format
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    # Only include tools if we have them
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"  # LLM decides whether to use a tool

    # HTTP headers — standard auth pattern
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter-specific headers for tracking
        "HTTP-Referer": "http://localhost",
        "X-Title": "Research Agent",
    }

    # ─── MAKE THE API CALL ───────────────────────────────────────
    # httpx.AsyncClient() creates a connection pool that can be reused.
    # The `timeout=60` prevents hanging forever if the API is slow.
    # LLMs can take 5-30 seconds to respond, especially with long contexts.

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload
        )

        # Check for API errors
        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(
                f"LLM API error (HTTP {response.status_code}): {error_detail}"
            )

        data = response.json()

    # ─── PARSE THE RESPONSE ──────────────────────────────────────
    # The response has this structure:
    # {
    #   "choices": [{
    #     "message": {
    #       "role": "assistant",
    #       "content": "text response" or null,
    #       "tool_calls": [{ "id": "...", "function": { "name": "search", "arguments": "{...}" } }]
    #     },
    #     "finish_reason": "stop" or "tool_calls"
    #   }]
    # }

    choice = data["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason", "stop")

    result = {
        "content": message.get("content", ""),
        "tool_calls": [],
        "finish_reason": finish_reason,
        "raw_message": message,  # keep the full message for appending to history
    }

    # Parse tool calls if present
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            result["tool_calls"].append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"]),
            })

    return result
