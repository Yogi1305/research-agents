"""
╔══════════════════════════════════════════════════════════════════╗
║                  SEARCH TOOL — Serper.dev                        ║
║                                                                  ║
║  This is a "tool" — a function the LLM can request to call.      ║
║  The LLM says "I want to search for X" and this code does it.    ║
╚══════════════════════════════════════════════════════════════════╝

=== CONCEPT: TOOL USE (FUNCTION CALLING) ===

"Tool use" is what separates a chatbot from an agent.

A chatbot can only:
  - Read your question
  - Generate text based on its training data
  - That's it — it can't look anything up, run code, or take actions

An agent can:
  - Read your question
  - DECIDE it needs more information
  - REQUEST your code to call a function (search, read, calculate, etc.)
  - RECEIVE the function result
  - Use that result to make its next decision

The flow for this search tool:

  1. LLM returns: {"name": "search", "arguments": {"query": "climate change effects"}}
  2. Our Python code catches this and calls: search("climate change effects")
  3. This function calls the Serper API (a Google Search wrapper)
  4. We get back real search results
  5. We send those results back to the LLM
  6. The LLM reads them and decides: read a page? search more? finish?

The LLM NEVER calls Serper directly. It just says "I want to search."
Your code does the actual work.

=== CONCEPT: WHY SERPER.DEV? ===

Why not just scrape Google directly?
  - Google blocks scrapers and requires CAPTCHAs
  - Rate limits are unpredictable
  - HTML parsing is fragile (Google changes layout frequently)

Serper.dev:
  - Simple REST API: one POST request → structured JSON results
  - Free tier: 2,500 searches (plenty for learning)
  - Returns clean data: title, snippet, URL — no HTML parsing needed
  - Response time: ~200ms (fast!)

Alternative search APIs:
  - SerpAPI ($50/mo) — more features, expensive
  - Tavily ($0 free tier) — designed for AI agents specifically
  - Brave Search API — privacy-focused, free tier
  - Google Custom Search — official but limited (100 free/day)
"""

import httpx
import os
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


async def search(query: str) -> list[dict]:
    """
    Search the web using Serper.dev (Google Search API).

    Parameters
    ----------
    query : str
        The search query (e.g., "What caused the 2008 financial crisis")

    Returns
    -------
    list[dict]
        Each dict has: title, url, snippet
        Returns up to 10 results

    Example return value:
    [
        {
            "title": "The 2008 Financial Crisis Explained",
            "url": "https://example.com/...",
            "snippet": "The crisis was caused by..."
        },
        ...
    ]

    === WHY WE RETURN A SIMPLIFIED FORMAT ===

    The raw Serper response has a LOT of data we don't need (ads,
    related searches, knowledge panels, etc.). We extract only what
    the LLM needs to decide which pages to read:
      - title:   what's the page about?
      - url:     where to read it
      - snippet: a preview — is this relevant?

    This is "context management" — don't waste the LLM's attention
    on irrelevant data. Every token counts.
    """
    if not SERPER_API_KEY:
        raise ValueError(
            "SERPER_API_KEY not found!\n"
            "Get a free key from https://serper.dev"
        )

    # ─── CALL THE SERPER API ─────────────────────────────────────
    # Serper uses POST with a JSON body (not query params like most search APIs)
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": 10  # number of results (max 100, but 10 is plenty)
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            SERPER_URL,
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise RuntimeError(f"Serper API error: {response.status_code} — {response.text}")

        data = response.json()

    # ─── EXTRACT CLEAN RESULTS ───────────────────────────────────
    # The raw response has "organic" results (the main search results)
    # plus other sections we don't need.
    results = []
    for item in data.get("organic", []):
        results.append({
            "title": item.get("title", "No title"),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", "No snippet available"),
        })

    return results
