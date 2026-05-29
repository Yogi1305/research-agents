"""
╔══════════════════════════════════════════════════════════════════╗
║                   READ TOOL — Jina Reader                        ║
║                                                                  ║
║  Reads the full text content of any webpage.                     ║
║  The LLM decides WHICH page to read based on search results.     ║
╚══════════════════════════════════════════════════════════════════╝

=== CONCEPT: WHY DO WE NEED A "READ" TOOL? ===

The search tool returns snippets — tiny previews.
Those aren't enough to answer complex questions.

After searching, the LLM looks at the snippets and decides
"this page looks promising, let me read the full content."
That's what this tool does — fetches the full page text.

=== CONCEPT: WHY JINA READER? ===

Normally, to extract text from a webpage you'd need to:
  1. Fetch the HTML with httpx
  2. Parse it with BeautifulSoup
  3. Strip out navigation, ads, headers, footers
  4. Extract just the article content
  5. Handle JavaScript-rendered pages (SPAs) with Playwright

That's a LOT of code for every project.

Jina Reader does it for free with NO setup:
  https://r.jina.ai/https://example.com

Just prefix any URL with "https://r.jina.ai/" and you get:
  - Clean, readable text (no HTML tags)
  - JavaScript-rendered content handled automatically
  - Free, no API key needed
  - Returns markdown-formatted text

It's the simplest way to read web pages for AI agents.

=== CONCEPT: TOKEN CONTEXT MANAGEMENT ===

LLMs have a "context window" — a maximum number of tokens they
can process at once. For example:
  - GPT-4:      128,000 tokens (~96,000 words)
  - Claude 3.5: 200,000 tokens (~150,000 words)
  - Mistral 7B:   8,000 tokens (~6,000 words)

A single webpage can be 10,000+ words. If the agent reads 5 pages,
that's 50,000+ words — which could EXCEED the context window.

Solutions:
  1. TRIM: cut page content to a max length (what we do here)
  2. SUMMARIZE: use a separate LLM call to condense the page
  3. CHUNK: split into sections and only keep relevant ones

We use trimming (5,000 characters ≈ 1,250 words) as the simplest
approach. It works well because important content is usually near
the top of articles.
"""

import httpx


# Prefix any URL with this to get readable text
JINA_PREFIX = "https://r.jina.ai/"

# Maximum characters to return from a page
# 5000 chars ≈ 1250 words ≈ ~1500 tokens
# This prevents any single page from consuming too much context
MAX_CONTENT_LENGTH = 5000


async def read(url: str) -> str:
    """
    Read the full text content of a webpage using Jina Reader.

    Parameters
    ----------
    url : str
        The URL to read (e.g., "https://en.wikipedia.org/wiki/...")

    Returns
    -------
    str
        The plain text content of the page, trimmed to MAX_CONTENT_LENGTH

    How it works:
        1. We prepend "https://r.jina.ai/" to the URL
        2. Jina fetches the page, renders JavaScript, extracts text
        3. We get back clean, readable text (no HTML)
        4. We trim it to avoid token overflow

    Error handling:
        If the page fails to load, we return an error message
        instead of crashing. The LLM will see this and try a
        different URL.
    """
    try:
        # ─── FETCH VIA JINA READER ───────────────────────────────
        # The magic: just prefix ANY URL with r.jina.ai/
        # No API key, no setup, no parsing — just clean text.
        jina_url = f"{JINA_PREFIX}{url}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                jina_url,
                headers={
                    "Accept": "text/plain",  # request plain text format
                },
                follow_redirects=True  # follow any URL redirects
            )

            if response.status_code != 200:
                return f"Error: Could not read page (HTTP {response.status_code})"

            content = response.text

        # ─── TRIM CONTENT ────────────────────────────────────────
        # Why trim?
        #   1. Prevents token overflow in the LLM context
        #   2. Reduces cost (more tokens = more money)
        #   3. Important info is usually at the top of articles
        #
        # If content is longer than our limit, we add a note so
        # the LLM knows it's seeing a trimmed version.
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "\n\n[Content trimmed — page was too long]"

        return content

    except httpx.TimeoutException:
        return "Error: Page took too long to load (timeout)"
    except Exception as e:
        return f"Error reading page: {str(e)}"
