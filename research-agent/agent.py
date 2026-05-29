"""
╔══════════════════════════════════════════════════════════════════╗
║                     THE AGENT LOOP                               ║
║                                                                  ║
║  This is the HEART of any AI agent.                              ║
║  Understanding this file = understanding agentic AI.             ║
╚══════════════════════════════════════════════════════════════════╝

=== CONCEPT: WHAT IS AN AGENT LOOP? ===

A chatbot: you ask → it responds → done.
An agent:  you ask → it THINKS → it ACTS → it OBSERVES → it THINKS again → ...

This is called the "ReAct" pattern (Reason + Act):

  ┌─────────┐
  │  THINK  │  ← LLM decides what to do next
  └────┬────┘
       │
  ┌────▼────┐
  │   ACT   │  ← Call a tool (search, read, etc.)
  └────┬────┘
       │
  ┌────▼────┐
  │ OBSERVE │  ← Get the tool result
  └────┬────┘
       │
       └──────→ Back to THINK (loop continues)

The loop continues until:
  1. The agent calls finish() — it has enough info
  2. Max iterations reached — safety limit to prevent infinite loops
  3. The LLM responds without a tool call — unusual but possible

=== CONCEPT: WHY IS THIS A "LOOP" AND NOT A "CHAIN"? ===

A chain: Step 1 → Step 2 → Step 3 → Done  (predetermined)
A loop:  Step ? → Step ? → Step ? → Done  (agent decides)

In a chain, YOU decide the steps in advance.
In a loop, the LLM decides each step dynamically based on what
it has learned so far.

This is what makes it an "agent" — it has AUTONOMY.

=== CONCEPT: CONVERSATION HISTORY AS MEMORY ===

The LLM is stateless — it doesn't remember previous calls.
Every time we call it, we send the ENTIRE conversation history.

  Call 1: [user question]
  Call 2: [user question, assistant thinking, tool result]
  Call 3: [user question, assistant thinking, tool result, assistant thinking, tool result]

The history grows with each iteration. This IS the agent's memory.
The LLM reads the full history and decides what to do next.

=== CONCEPT: STOPPING CONDITIONS ===

Without stopping conditions, an agent loops forever. We use TWO:

  1. EXPLICIT STOP: The agent calls the finish() tool
     → This is the "happy path" — agent found what it needs

  2. ITERATION LIMIT: Maximum 10 loops, then force-stop
     → This is the "safety valve" — prevents runaway agents
     → Also prevents API cost from spiraling

Both are essential. Never build an agent without an iteration limit.
"""

import json
from datetime import datetime

from llm import call_llm
from tools.search import search
from tools.read import read
from prompts.researcher import SYSTEM_PROMPT, TOOLS


async def run_agent(question: str, verbose: bool = True) -> dict:
    """
    Run the research agent on a question.

    This function IS the agent loop. Here's what happens:

    1. We start with the user's question
    2. We call the LLM with the question + tool definitions
    3. The LLM either:
       a) Requests a tool call → we execute it, add result, loop back
       b) Calls finish() → we return the answer
       c) Just responds with text → we return that
    4. Repeat until finish() or max iterations

    Parameters
    ----------
    question : str
        The user's research question
    verbose : bool
        If True, print each step to console (great for learning!)

    Returns
    -------
    dict with:
        - answer: str — the final answer
        - sources: list[str] — URLs used
        - iterations: int — how many loops it took
        - history: list — full conversation history (for debugging)
    """

    # ─── INITIALIZE CONVERSATION HISTORY ─────────────────────────
    # The messages list IS the agent's memory.
    # We start with the system prompt (who the agent is) and the
    # user's question.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # ─── TRACKING ────────────────────────────────────────────────
    max_iterations = 10      # Safety limit — NEVER remove this
    sources_read = []        # Track which URLs were read
    search_count = 0         # Track number of searches performed
    read_count = 0           # Track number of pages read

    if verbose:
        print(f"\n{'='*60}")
        print(f"🔍 RESEARCH AGENT")
        print(f"{'='*60}")
        print(f"Question: {question}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════
    # THE AGENT LOOP — This is where the magic happens
    # ═══════════════════════════════════════════════════════════════
    #
    # Each iteration:
    #   1. Send full conversation history + tools to LLM
    #   2. LLM decides: call a tool, or finish
    #   3. If tool call: execute it, add result to history, continue
    #   4. If finish: return the answer
    #

    for iteration in range(1, max_iterations + 1):
        if verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"── Iteration {iteration}/{max_iterations} [{timestamp}] ──")

        # ─── STEP 1: ASK THE LLM ────────────────────────────────
        # We send the FULL history every time.
        # The LLM reads everything and decides what to do next.
        response = await call_llm(messages=messages, tools=TOOLS)

        # ─── STEP 2: CHECK IF THE LLM WANTS TO USE A TOOL ───────
        # The response tells us what the LLM wants to do:
        #   - tool_calls present → it wants to call one or more tools
        #   - no tool_calls     → it just wants to respond with text

        if response["tool_calls"]:
            # ─── THE LLM WANTS TO CALL A TOOL ───────────────────
            # First, add the assistant's message (with tool call info)
            # to the conversation history
            messages.append(response["raw_message"])

            # Process EACH tool call the LLM requested
            # (Most models call one tool at a time, but some can
            #  call multiple in parallel — we handle both)
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_id = tool_call["id"]

                if verbose:
                    print(f"  🔧 Tool: {tool_name}({json.dumps(tool_args, indent=2)})")

                # ─── STEP 3: EXECUTE THE TOOL ────────────────────
                # This is where YOUR CODE does the actual work.
                # The LLM asked for it, we execute it.

                if tool_name == "search":
                    # ─── SEARCH TOOL ─────────────────────────────
                    search_count += 1
                    result = await search(tool_args["query"])

                    # Format results for the LLM to read
                    # We convert the list of dicts into readable text
                    formatted = "\n\n".join([
                        f"**{r['title']}**\n"
                        f"URL: {r['url']}\n"
                        f"Snippet: {r['snippet']}"
                        for r in result
                    ])
                    tool_result = f"Found {len(result)} results:\n\n{formatted}"

                    if verbose:
                        print(f"  📋 Found {len(result)} search results")

                elif tool_name == "read":
                    # ─── READ TOOL ───────────────────────────────
                    read_count += 1
                    url = tool_args["url"]
                    sources_read.append(url)
                    tool_result = await read(url)

                    if verbose:
                        preview = tool_result[:100].replace('\n', ' ')
                        print(f"  📄 Read {len(tool_result)} chars from {url}")
                        print(f"     Preview: {preview}...")

                elif tool_name == "finish":
                    # ─── FINISH TOOL ─────────────────────────────
                    # The agent is done! Extract the answer.
                    if verbose:
                        print(f"\n  ✅ Agent finished after {iteration} iteration(s)")
                        print(f"     Searches: {search_count}, Pages read: {read_count}")
                        print(f"     Sources: {tool_args.get('sources', [])}")

                    return {
                        "answer": tool_args["answer"],
                        "sources": tool_args.get("sources", []),
                        "iterations": iteration,
                        "searches": search_count,
                        "pages_read": read_count,
                    }
                else:
                    # Unknown tool — shouldn't happen, but handle gracefully
                    tool_result = f"Error: Unknown tool '{tool_name}'"

                # ─── STEP 4: FEED RESULT BACK TO THE LLM ────────
                # We add the tool result as a "tool" message.
                # The LLM will read this on the next iteration
                # and decide what to do next.
                #
                # The tool_call_id links this result to the specific
                # tool call that requested it (important when multiple
                # tools are called in parallel).
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_result,
                })

        else:
            # ─── NO TOOL CALL — LLM JUST RESPONDED ──────────────
            # This happens when the LLM decides to answer directly
            # without using tools. Unusual for a research agent,
            # but we handle it gracefully.
            if verbose:
                print(f"  💬 LLM responded without a tool call")

            return {
                "answer": response["content"],
                "sources": sources_read,
                "iterations": iteration,
                "searches": search_count,
                "pages_read": read_count,
            }

        if verbose:
            print()  # blank line between iterations

    # ─── MAX ITERATIONS REACHED ──────────────────────────────────
    # If we get here, the agent didn't finish within the limit.
    # This is the safety valve — it prevents infinite loops and
    # runaway API costs.
    if verbose:
        print(f"\n  ⚠️  Max iterations ({max_iterations}) reached!")
        print(f"  The agent could not find a confident answer in time.")

    return {
        "answer": (
            f"I was unable to find a fully confident answer within {max_iterations} iterations. "
            f"Here's what I found so far based on {read_count} pages I read from {search_count} searches."
        ),
        "sources": sources_read,
        "iterations": max_iterations,
        "searches": search_count,
        "pages_read": read_count,
    }
