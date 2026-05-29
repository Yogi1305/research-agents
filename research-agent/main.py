"""
╔══════════════════════════════════════════════════════════════════╗
║                   MAIN — Entry Point                             ║
║                                                                  ║
║  Run this file to use the research agent.                        ║
║                                                                  ║
║  Usage:                                                          ║
║    python main.py                         (interactive mode)     ║
║    python main.py "your question here"    (single question)      ║
╚══════════════════════════════════════════════════════════════════╝

=== CONCEPT: async/await IN PYTHON ===

Why is this code async?

Normal (synchronous) code:
  result1 = requests.get(url1)    # waits 2 seconds
  result2 = requests.get(url2)    # waits 2 seconds
  # Total: 4 seconds

Async code:
  result1 = await httpx.get(url1)  # starts request, lets other code run
  result2 = await httpx.get(url2)  # starts request, lets other code run
  # Can overlap! Total: ~2 seconds

For this project (single agent, sequential), the benefit is small.
But in Project 5, where 4 agents run IN PARALLEL, async is essential.

We use async from the start so you build the muscle memory.

Key terms:
  - `async def` → this function can use `await`
  - `await` → "start this task, but don't block — come back when done"
  - `asyncio.run()` → starts the async event loop (you call this once)

=== CONCEPT: WHY IS main.py SEPARATE FROM agent.py? ===

Separation of concerns:
  - main.py → user interface (how you interact with it)
  - agent.py → agent logic (how the agent works)
  - llm.py → API communication (how we talk to the LLM)
  - tools/ → tool implementations (what the agent can do)

Each file has ONE job. This makes it easy to:
  - Swap the interface (CLI → web → API)
  - Change the LLM provider
  - Add/remove tools
  - Test components independently
"""

import asyncio
import sys
import json
from datetime import datetime

from agent import run_agent


def print_result(result: dict):
    """Pretty-print the agent's result."""
    print(f"\n{'='*60}")
    print(f"📝 RESEARCH RESULT")
    print(f"{'='*60}\n")

    # Print the answer
    print(result["answer"])

    # Print sources
    if result["sources"]:
        print(f"\n{'─'*40}")
        print(f"📚 Sources ({len(result['sources'])}):")
        for i, source in enumerate(result["sources"], 1):
            print(f"  {i}. {source}")

    # Print stats
    print(f"\n{'─'*40}")
    print(f"📊 Stats:")
    print(f"  • Iterations: {result['iterations']}")
    print(f"  • Searches performed: {result['searches']}")
    print(f"  • Pages read: {result['pages_read']}")
    print(f"{'='*60}\n")


async def interactive_mode():
    """
    Run the agent in interactive mode — ask questions in a loop.

    Type your question and press Enter.
    Type 'quit' or 'exit' to stop.
    """
    print(f"\n{'='*60}")
    print(f"🤖 RESEARCH AGENT — Interactive Mode")
    print(f"{'='*60}")
    print(f"Ask me anything! I'll search the web and find answers.")
    print(f"Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("❓ Your question: ").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 👋")
                break

            # Run the agent
            result = await run_agent(question, verbose=True)
            print_result(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again with a different question.\n")


async def single_question(question: str):
    """Run the agent on a single question from command line args."""
    result = await run_agent(question, verbose=True)
    print_result(result)


def main():
    """
    Entry point.

    === HOW asyncio.run() WORKS ===

    asyncio.run() does three things:
      1. Creates an event loop (the async "engine")
      2. Runs our async function inside it
      3. Cleans up when done

    You only call asyncio.run() ONCE — at the top level.
    Everything else uses `await`.
    """

    if len(sys.argv) > 1:
        # Question passed as command line argument
        question = " ".join(sys.argv[1:])
        asyncio.run(single_question(question))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
