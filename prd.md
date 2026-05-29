# Agentic AI — Project PRDs (Python Edition)
### Learning Path: From Tool Use to Production-Grade Multi-Agent Systems

---

## How to Read These PRDs

Each PRD is structured the same way:
- **What you're building** — plain English description
- **What you'll learn** — the actual concepts, not just buzzwords
- **Architecture** — how the system is laid out
- **Agent design** — roles, responsibilities, prompt structure
- **Data flow** — step by step what happens when you run it
- **Tech stack** — exact tools and why
- **Acceptance criteria** — how you know you're done
- **Checkpoints** — sub-goals to hit before moving on

Work through these in order. Each project builds on the previous one.

---

## Python Baseline (applies to all 5 projects)

**Python version:** 3.11+

**Every project uses these:**
```
pip install httpx python-dotenv pydantic
```

**Every project has this structure at minimum:**
```
project-name/
├── .env                  ← API keys, never commit this
├── requirements.txt      ← pip dependencies
├── main.py               ← entry point
```

**.env format:**
```
OPENROUTER_API_KEY=sk-or-...
SERPER_API_KEY=...
```

**Loading env in every file:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
```

---

---

# PROJECT 1: Research Agent

## Plain English

You ask it a question. It searches the web, reads the pages it finds, decides if it has enough information, and writes you a structured answer with sources. It does this entirely on its own — you don't tell it which pages to read or when to stop.

---

## What You'll Learn

| Concept | What it means in practice |
|---|---|
| Tool use | Defining functions the LLM can call, and handling when it calls them |
| The agent loop | The repeating cycle: think → act → observe → think again |
| Stopping conditions | Teaching an agent when it has "enough" information |
| Prompt engineering for agents | System prompts that give an agent a role, tools, and rules |
| Structured LLM output | Getting the model to return JSON so your code can parse the decision |
| Token context management | What to do when the context gets too long |

---

## Architecture

```
User Input (question)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │              RESEARCH AGENT                 │
  │                                             │
  │  System Prompt: You are a researcher.       │
  │  You have 2 tools: search() and read()      │
  │  Decide when you have enough info.          │
  │                                             │
  │  ┌──────────────────────────────────────┐   │
  │  │           AGENT LOOP                 │   │
  │  │                                      │   │
  │  │  LLM decides next action:            │   │
  │  │  - call search(query)                │   │
  │  │  - call read(url)                    │   │
  │  │  - call finish(answer)               │   │
  │  │                                      │   │
  │  │  Tool result appended to context     │   │
  │  │  Loop continues until finish()       │   │
  │  └──────────────────────────────────────┘   │
  └─────────────────────────────────────────────┘
       │
       ▼
  Final Answer + Sources
```

---

## Agent Design

### Single Agent: The Researcher

**System prompt:**
```
You are a research agent. Your job is to answer the user's question accurately.

You have the following tools:
- search(query: str) — returns a list of web results (title, url, snippet)
- read(url: str) — returns the full text of a webpage
- finish(answer: str, sources: list[str]) — call this when confident in your answer

Rules:
1. Always search before reading
2. Read at least 2 sources before finishing
3. If a page is irrelevant, do not read more from it
4. Never make up information — only use what tools return
5. When you call finish(), your answer must cite specific sources
```

**Tool definitions (what you pass to the model):**
```python
TOOLS = [
    {
        "name": "search",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query" }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read",
        "description": "Read the full content of a webpage",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to read" }
            },
            "required": ["url"]
        }
    },
    {
        "name": "finish",
        "description": "Call this when you have enough information to answer",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": { "type": "string" },
                "sources": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["answer", "sources"]
        }
    }
]
```

---

## Data Flow

```
1. User inputs: "What caused the 2008 financial crisis?"

2. Agent loop iteration 1:
   LLM response: tool_call → search(query="causes of 2008 financial crisis")
   Your code calls Serper API
   Returns: list of 10 results (titles + snippets + URLs)
   Appended to messages as: role="tool", content=[results]

3. Agent loop iteration 2:
   LLM response: tool_call → read(url="https://...")
   Your code calls r.jina.ai/{url}
   Returns: page content (trimmed to 3000 tokens)
   Appended to messages

4. Agent loop iteration 3:
   LLM response: tool_call → read(url="https://...")
   (reads second source)

5. Agent loop iteration 4:
   LLM response: tool_call → finish(answer="...", sources=["url1","url2"])
   Loop exits. Output printed.
```

---

## The Agent Loop in Python

```python
import httpx
import json

async def run_agent(question: str):
    messages = [{"role": "user", "content": question}]
    max_iterations = 10

    for i in range(max_iterations):
        response = await call_llm(messages, TOOLS)
        
        # Check what the LLM wants to do
        if response["stop_reason"] == "tool_use":
            tool_call = extract_tool_call(response)
            
            if tool_call["name"] == "search":
                result = await search(tool_call["input"]["query"])
            elif tool_call["name"] == "read":
                result = await read(tool_call["input"]["url"])
            elif tool_call["name"] == "finish":
                # Done — return the answer
                return tool_call["input"]
            
            # Append tool result back to conversation
            messages.append({"role": "assistant", "content": response["content"]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_call["id"], "content": str(result)}
            ]})
        else:
            # LLM responded without a tool call (shouldn't happen often)
            return {"answer": response["content"], "sources": []}

    return {"answer": "Could not find a confident answer within iteration limit.", "sources": []}
```

---

## Tech Stack

| Tool | Purpose | Why |
|---|---|---|
| OpenRouter | LLM API | Access to any model |
| `httpx` | Async HTTP client | Better than requests for async code |
| `python-dotenv` | Load .env file | Standard practice |
| `pydantic` | Data validation | Validate tool call arguments |
| Serper.dev | Search API | Free tier, simple REST |
| Jina Reader (`r.jina.ai`) | Web page reader | Free, no setup — prefix any URL |

**Recommended model:** `mistralai/mistral-7b-instruct` (cheap, fast, good at tool use)

**Jina Reader trick:** To read any URL for free:
```python
async def read(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://r.jina.ai/{url}")
        return response.text[:3000]  # trim to avoid token overflow
```

---

## File Structure

```
research-agent/
├── main.py               ← entry point, takes user question
├── agent.py              ← the agent loop
├── tools/
│   ├── __init__.py
│   ├── search.py         ← calls Serper API
│   └── read.py           ← calls Jina Reader
├── prompts/
│   └── researcher.py     ← system prompt + tool definitions
├── requirements.txt
└── .env
```

**requirements.txt:**
```
httpx
python-dotenv
pydantic
```

---

## Acceptance Criteria

1. Agent correctly answers 5 different factual questions without any help from you
2. Agent reads at least 2 sources before finishing every time
3. Agent never hallucinates (all claims traceable to tool results)
4. Agent terminates — it never runs more than 10 loop iterations
5. You can explain out loud what happens at each step of the loop

---

## Checkpoints

**Checkpoint 1:** Agent calls `search()` and you print the raw tool call to console. You understand what the LLM returned and why.

**Checkpoint 2:** Agent loop runs — search, read, finish — and you print the full messages list at each step. You can read it like a conversation.

**Checkpoint 3:** Agent answers 3 questions correctly end-to-end.

**Checkpoint 4:** Add a max iteration limit. Agent gracefully stops at 10 and says "I couldn't find a confident answer."

---

---

# PROJECT 2: Planner + Executor (Two-Agent System)

## Plain English

User gives a vague, multi-step goal. A **Planner agent** thinks through what needs to be done and produces a structured task list. An **Executor agent** works through each task one by one, producing output for each. The Planner and Executor never talk to each other directly — your code coordinates them.

---

## What You'll Learn

| Concept | What it means in practice |
|---|---|
| Multi-agent orchestration | Your code as the "manager" between agents |
| Structured JSON output | Forcing an LLM to return machine-readable task lists |
| Context passing between agents | How to give Agent B the output of Agent A |
| Role differentiation | Why a planner model ≠ an executor model |
| Multi-model usage | Using a smarter model for planning, cheaper for execution |
| Task dependency | What to do when task 3 needs task 2's output |
| Pydantic validation | Validating LLM output against a schema |

---

## Architecture

```
User Goal (natural language)
       │
       ▼
  ┌────────────────────┐
  │   PLANNER AGENT    │
  │   (smarter model)  │
  │                    │
  │   Input: goal      │
  │   Output: JSON     │
  │   task list        │
  └────────────────────┘
       │
       │  [ { id, title, description, depends_on } ]
       ▼
  ┌────────────────────────────────────────┐
  │      ORCHESTRATOR (your code)          │
  │                                        │
  │  for each task in plan:                │
  │    gather context from prior outputs   │
  │    call Executor with task + context   │
  │    store output                        │
  └────────────────────────────────────────┘
       │
       ▼
  ┌────────────────────┐
  │   EXECUTOR AGENT   │
  │  (cheaper model)   │
  │                    │
  │  Input: one task   │
  │  + prior outputs   │
  │  Output: task      │
  │  result (text)     │
  └────────────────────┘
       │
       ▼
  Assembled Final Output
```

---

## Agent Design

### Agent 1: The Planner

**System prompt:**
```
You are a project planner. Given a goal, break it into a list of concrete tasks.

Return ONLY a JSON array. No explanation, no markdown, no preamble.
Each task has:
- id: integer (starting from 1)
- title: short task name
- description: what needs to be done in this task
- depends_on: list of task IDs this task needs before running (empty list if none)

Rules:
- Maximum 7 tasks
- Tasks must be executable independently or after their dependencies
- Be specific — "research X" is not enough, say what to look for
```

**Pydantic schema for validation:**
```python
from pydantic import BaseModel

class Task(BaseModel):
    id: int
    title: str
    description: str
    depends_on: list[int]

class TaskList(BaseModel):
    tasks: list[Task]
```

### Agent 2: The Executor

**System prompt:**
```
You are a focused task executor. You will be given:
1. A specific task to complete
2. Outputs from previous tasks (if relevant)

Complete only the task given. Be thorough but do not do more than asked.
Return your output as plain text or markdown as appropriate.
```

---

## Data Flow

```
1. User input: "Write a competitive analysis of Notion vs Linear"

2. Planner called:
   Input: user goal
   Output: JSON task list (5 tasks)
   Your code: json.loads() → pydantic validate → Task list

3. Orchestrator logic:
   Sort tasks by dependency order (topological sort)
   Task 1 (no deps) → run immediately
   Task 2 (no deps) → run immediately
   Task 3 (depends on [1,2]) → wait for both, then run with their outputs

4. Executor called per task:
   Input: { "task": {...}, "previous_outputs": { "1": "...", "2": "..." } }
   Output: task result text
   Stored in: outputs[task.id]

5. Final step:
   All outputs assembled into final document
```

---

## Topological Sort in Python

```python
def sort_by_dependency(tasks: list[Task]) -> list[Task]:
    """Returns tasks in safe execution order respecting depends_on."""
    sorted_tasks = []
    completed_ids = set()
    remaining = list(tasks)

    while remaining:
        ready = [t for t in remaining if all(dep in completed_ids for dep in t.depends_on)]
        if not ready:
            raise ValueError("Circular dependency detected in task plan")
        for task in ready:
            sorted_tasks.append(task)
            completed_ids.add(task.id)
            remaining.remove(task)

    return sorted_tasks
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| OpenRouter | LLM API |
| `httpx` | Async HTTP |
| `pydantic` | Validate Planner JSON output |
| `python-dotenv` | Env vars |
| `anthropic/claude-3-haiku` | Planner (strong JSON output) |
| `mistralai/mistral-7b-instruct` | Executor (cheap, runs per task) |

---

## File Structure

```
planner-executor/
├── main.py                  ← entry point
├── orchestrator.py          ← dependency resolution + loop
├── agents/
│   ├── __init__.py
│   ├── planner.py           ← calls LLM, returns validated TaskList
│   └── executor.py          ← takes one task + context, returns result
├── models/
│   └── task.py              ← pydantic Task and TaskList models
├── prompts/
│   ├── planner.py
│   └── executor.py
├── utils/
│   └── sort.py              ← topological sort
├── requirements.txt
└── .env
```

**requirements.txt:**
```
httpx
python-dotenv
pydantic
```

---

## Acceptance Criteria

1. Planner produces valid, schema-conforming JSON for 5 different goal types
2. Orchestrator correctly runs tasks in dependency order
3. Executor receives and uses previous task outputs
4. Final output is coherent and clearly built from task chain
5. System handles Planner producing invalid JSON gracefully (retry once, then raise)
6. You can explain why task 3 ran after tasks 1 and 2

---

## Checkpoints

**Checkpoint 1:** Planner returns valid JSON for 3 different goals. Pydantic validates it without error.

**Checkpoint 2:** Print execution order before running. Confirm it's correct for a goal with at least one dependency.

**Checkpoint 3:** Executor correctly uses prior outputs — test this by having task 3 explicitly reference something from task 1's output.

**Checkpoint 4:** Run end-to-end on a real goal. Print each task output before assembling final result.

---

---

# PROJECT 3: Knowledge Base Agent (Memory Layer)

## Plain English

You have a folder of text files — notes, documents, articles. The agent can answer questions about them, retrieve specific information, and tell you when it doesn't know something. It uses vector embeddings to find relevant content instead of reading every file on every query.

---

## What You'll Learn

| Concept | What it means in practice |
|---|---|
| Embeddings | Turning text into numbers that represent meaning |
| Vector similarity search | Finding text semantically close to a query |
| RAG (Retrieval Augmented Generation) | Retrieve relevant chunks → give to LLM → get answer |
| Chunking strategy | How to split documents so retrieval works well |
| ChromaDB | Local vector database, persisted to disk |
| Grounding | Ensuring the LLM only answers from retrieved content |
| Persistent memory | Data that survives between sessions |

---

## Architecture

```
INGESTION PIPELINE (run once per document set):

  document.txt
       │
       ▼
  ┌────────────────┐
  │    CHUNKER     │  splits text into ~500 token overlapping chunks
  └────────────────┘
       │
       ▼
  ┌────────────────┐
  │   EMBEDDER     │  calls embedding model → vector per chunk
  └────────────────┘
       │
       ▼
  ┌────────────────┐
  │   ChromaDB     │  stores (chunk_text, vector, metadata)
  │  (on disk)     │
  └────────────────┘


QUERY PIPELINE (runs on each question):

  User Question
       │
       ▼
  Embed question → get query vector
       │
       ▼
  ChromaDB similarity search → top 5 matching chunks
       │
       ▼
  ┌────────────────────────────────────┐
  │            QA AGENT                │
  │                                    │
  │  Context: [5 retrieved chunks]     │
  │  Question: user's question         │
  │  Rule: only answer from context    │
  │  If not in context → say so        │
  └────────────────────────────────────┘
       │
       ▼
  Answer + source references
```

---

## Agent Design

### Single Agent: The Knowledge Agent

**System prompt:**
```python
def build_prompt(retrieved_chunks: list[dict], question: str) -> str:
    context = "\n\n---\n\n".join([
        f"Source: {chunk['metadata']['filename']}\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])
    return f"""You are a knowledge assistant. Answer using ONLY the context below.
If the answer is not in the context, say: "I don't have information about that in my knowledge base."
Always cite which source document your answer comes from.

CONTEXT:
{context}

QUESTION: {question}"""
```

No tool use in this project — retrieval happens in your Python code before the LLM is ever called.

---

## The Chunker

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks.
    Overlap ensures context at chunk boundaries is not lost.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward but keep overlap

    return chunks
```

---

## Data Flow

### Ingestion (run once):
```
1. Read all .txt / .md files from /knowledge directory
2. For each file:
   a. chunk_text() → list of ~500 word chunks
   b. For each chunk: call embedding API → 1536-dim vector
   c. chromadb.add(id, text, vector, metadata={filename, chunk_index})
3. ChromaDB persists to ./chroma_db/ on disk
```

### Query (runs every time):
```
1. User asks: "What does the document say about pricing strategy?"
2. Embed question → query vector
3. chroma_collection.query(query_vector, n_results=5) → top 5 chunks
4. Build prompt with retrieved chunks
5. Call LLM → answer
6. Return answer + source filenames
```

---

## Tech Stack

| Tool | Purpose | Why |
|---|---|---|
| OpenRouter | LLM for answering | |
| `chromadb` | Local vector database | Python-native, persists to disk, no server needed |
| `openai` SDK | Embedding model | `text-embedding-3-small` is cheap and good |
| `httpx` | HTTP calls | |
| `python-dotenv` | Env vars | |

**Note on embeddings:** You need an OpenAI API key just for embeddings (`text-embedding-3-small`). This is separate from OpenRouter. Alternatively use a free local embedding model via Ollama (`nomic-embed-text`).

---

## File Structure

```
knowledge-base-agent/
├── main.py               ← entry point (query mode)
├── ingest.py             ← run once to build the vector store
├── agent.py              ← builds prompt, calls LLM, returns answer
├── vector_store.py       ← ChromaDB wrapper (add, search)
├── chunker.py            ← text splitting logic
├── embedder.py           ← calls embedding API
├── knowledge/            ← your .txt and .md files go here
│   ├── doc1.txt
│   └── doc2.md
├── chroma_db/            ← auto-created by ChromaDB, do not edit
├── requirements.txt
└── .env
```

**requirements.txt:**
```
chromadb
openai
httpx
python-dotenv
```

---

## Acceptance Criteria

1. Ingestion correctly processes at least 10 documents
2. Retrieval returns relevant chunks 80%+ of the time (test manually)
3. Agent correctly says "I don't know" when query is outside the knowledge base
4. Agent cites source document in every answer
5. You can explain the difference between keyword search and vector similarity search
6. You can explain what chunk overlap is and why it exists
7. ChromaDB persists between runs — ingestion runs once, querying works forever after

---

## Checkpoints

**Checkpoint 1:** Chunker splits a document correctly. Print chunks and verify overlap exists between consecutive chunks.

**Checkpoint 2:** Embed two similar sentences and two dissimilar sentences. Print their cosine similarity scores. Similar ones should score higher.

**Checkpoint 3:** Retrieval works. Ask a question, print the top 5 retrieved chunks, verify they are relevant.

**Checkpoint 4:** Full QA pipeline works end-to-end. Agent answers from context and cites source correctly.

---

---

# PROJECT 4: Code Review Agent (Multi-Agent Pipeline)

## Plain English

You paste a code snippet. Three agents run in sequence: a **Reviewer** finds issues, a **Fixer** produces corrected code, and a **Critic** validates whether the fix actually solved the problems. If the Critic rejects the fix, the loop runs again. The whole thing is autonomous.

---

## What You'll Learn

| Concept | What it means in practice |
|---|---|
| Sequential agent pipelines | Output of Agent A becomes input of Agent B |
| Structured handoffs | Passing pydantic-validated data between agents |
| Critique loops | Using a third agent to validate another's output |
| Role prompting | Getting very different behavior from the same model via prompting |
| Loop termination | Preventing infinite critique cycles |
| Pydantic for LLM output | Validating every agent's JSON response |

---

## Architecture

```
Code Input
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    PIPELINE LOOP                     │
│                                                      │
│  ┌──────────────┐                                    │
│  │   REVIEWER   │  → list[Issue]                     │
│  │   Agent      │    (pydantic validated)            │
│  └──────────────┘                                    │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                    │
│  │    FIXER     │  → FixResult(fixed_code,           │
│  │    Agent     │              changes_made)         │
│  └──────────────┘                                    │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                    │
│  │    CRITIC    │  → CriticResult(approved,          │
│  │    Agent     │      remaining_issues,             │
│  └──────────────┘      confidence)                   │
│         │                                            │
│   approved = True  → exit loop                       │
│   approved = False → Fixer runs again                │
│   iteration >= 3   → exit with best attempt          │
└──────────────────────────────────────────────────────┘
    │
    ▼
Final reviewed + fixed code + summary report
```

---

## Pydantic Models

```python
from pydantic import BaseModel
from typing import Literal

class Issue(BaseModel):
    id: int
    severity: Literal["critical", "major", "minor"]
    line: int | None
    category: Literal["bug", "security", "performance", "style", "logic"]
    description: str
    suggestion: str

class FixResult(BaseModel):
    fixed_code: str
    changes_made: list[str]

class CriticResult(BaseModel):
    approved: bool
    confidence: int           # 1-10
    resolved_issue_ids: list[int]
    remaining_issue_ids: list[int]
    new_issues_introduced: list[str]
    verdict: str
```

---

## Agent Design

### Agent 1: The Reviewer

**System prompt:**
```
You are a senior code reviewer. Analyze the provided code and return a JSON array of issues.

Return ONLY a JSON array. No explanation outside the JSON.

Each issue must have:
- id (integer)
- severity: "critical", "major", or "minor"
- line: line number or null
- category: "bug", "security", "performance", "style", or "logic"
- description: what is wrong
- suggestion: how to fix it

If there are no issues, return an empty array [].
```

### Agent 2: The Fixer

**System prompt:**
```
You are a code fixer. You receive:
1. Original code
2. A list of issues to fix
3. (On retry) Feedback from a critic on your previous fix attempt

Return ONLY JSON in this exact format:
{
  "fixed_code": "complete fixed code as a string",
  "changes_made": ["description of each change made"]
}

The fixed_code must be the COMPLETE file, not a diff or partial snippet.
```

### Agent 3: The Critic

**System prompt:**
```
You are a code critic. You receive:
1. Original code with its issues
2. The fixed code
3. The list of issue IDs that should have been fixed

Evaluate the fix and return ONLY JSON:
{
  "approved": true or false,
  "confidence": integer 1-10,
  "resolved_issue_ids": [list of issue IDs that were fixed],
  "remaining_issue_ids": [list of issue IDs still present],
  "new_issues_introduced": ["any new problems the fix created"],
  "verdict": "one sentence summary"
}
```

---

## Data Flow

```
1. User pastes Python code

2. Reviewer runs:
   Input: code string
   Output: list[Issue] (pydantic validated)

3. Fixer runs (iteration 1):
   Input: original code + list[Issue]
   Output: FixResult

4. Critic runs:
   Input: original code + issues + fixed code
   Output: CriticResult { approved: False, remaining_issue_ids: [1, 3] }

5. Not approved → Fixer runs again (iteration 2):
   Input: original code + issues + previous fix + critic feedback
   Output: new FixResult

6. Critic runs again:
   Output: CriticResult { approved: True, confidence: 9 }

7. Loop exits. Print final report.
```

---

## The Pipeline Loop

```python
async def run_pipeline(code: str) -> dict:
    # Step 1: Review
    issues = await reviewer.run(code)
    
    if not issues:
        return {"result": code, "message": "No issues found", "iterations": 0}

    best_fix = None
    max_iterations = 3

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Step 2: Fix
        fix = await fixer.run(code, issues, previous_fix=best_fix)
        best_fix = fix

        # Step 3: Critique
        critique = await critic.run(code, issues, fix)
        print(f"Critic verdict: {critique.verdict} (confidence: {critique.confidence}/10)")

        if critique.approved:
            print(f"✓ Approved after {iteration} iteration(s)")
            break

        if iteration == max_iterations:
            print("⚠ Max iterations reached. Returning best attempt.")

    return {
        "original_issues": [i.model_dump() for i in issues],
        "fixed_code": best_fix.fixed_code,
        "changes_made": best_fix.changes_made,
        "final_verdict": critique.verdict,
        "approved": critique.approved
    }
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| OpenRouter | LLM API |
| `pydantic` | Validate every agent's JSON output |
| `httpx` | Async HTTP calls |
| `python-dotenv` | Env vars |

No external APIs. Pure LLM pipeline.

---

## File Structure

```
code-review-agent/
├── main.py                   ← entry point, reads code from file or stdin
├── pipeline.py               ← orchestrates the 3-agent loop
├── agents/
│   ├── __init__.py
│   ├── reviewer.py
│   ├── fixer.py
│   └── critic.py
├── models/
│   └── schemas.py            ← all pydantic models
├── prompts/
│   ├── reviewer.py
│   ├── fixer.py
│   └── critic.py
├── utils/
│   └── parse_json.py         ← safe JSON parser with fallback
├── requirements.txt
└── .env
```

**requirements.txt:**
```
httpx
pydantic
python-dotenv
```

---

## Acceptance Criteria

1. Pipeline correctly identifies real bugs in test code you write
2. Fixer resolves at least 80% of issues on first pass
3. Critic correctly identifies when issues remain
4. Loop terminates on approval OR max iterations — never infinite
5. Output includes a clear report: original issues, what was fixed, what remains
6. All three agents produce pydantic-valid output on every run
7. You can explain why a 3-agent design is better than one agent doing all three steps

---

## Checkpoints

**Checkpoint 1:** Write a Python function with 5 deliberate bugs. Reviewer finds all 5.

**Checkpoint 2:** Fixer produces syntactically valid Python on every run (use `ast.parse()` to verify).

**Checkpoint 3:** Critic correctly rejects a fix that still has issues. Force this by intentionally giving Fixer a weak prompt.

**Checkpoint 4:** Full loop runs 2+ iterations and terminates. Print iteration count and verdict each time.

---

---

# PROJECT 5: Autonomous Research Reporter with Human-in-the-Loop

## Plain English

User gives a research topic. Multiple agents run **in parallel** to research different angles simultaneously. They write findings into a **shared JSON file on disk**. A Writer agent synthesizes everything into a structured report. At key points, the system **pauses and asks you for approval** before continuing. All state is **persisted to disk** — if you stop mid-run and restart, it picks up exactly where it left off.

---

## What You'll Learn

| Concept | What it means in practice |
|---|---|
| Parallel agent execution | `asyncio.gather()` — running multiple agents simultaneously |
| Shared memory (write) | Multiple agents writing to the same JSON file |
| Human-in-the-loop | Pausing execution, waiting for approval, resuming |
| Async state persistence | Saving to disk so the process is resumable |
| Failure handling | One agent failing must not abort the whole pipeline |
| Agent communication via memory | Agents influence each other through shared state, not direct calls |
| `asyncio` fundamentals | Task creation, gather, cancellation |

---

## Architecture

```
User: "Research topic: Impact of LLMs on software developer jobs"
       │
       ▼
  ┌────────────────────────────────────────────┐
  │         COORDINATOR (your code)            │
  │  Checks disk for existing session          │
  │  If new: spawns 4 parallel agents          │
  │  If resuming: checks status, continues     │
  └────────────────────────────────────────────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ AGENT 1 │   │ AGENT 2 │   │ AGENT 3 │   │ AGENT 4 │
  │Academic │   │Industry │   │Community│   │Economic │
  │Research │   │  News   │   │Sentiment│   │  Data   │
  └─────────┘   └─────────┘   └─────────┘   └─────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  sessions/       │
                    │  {topic}.json    │  ← shared memory on disk
                    └──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  HUMAN CHECKPOINT │
                    │  "Review findings │
                    │   before writing" │
                    └─────────┬─────────┘
                              │ user types "yes" / edits file / "abort"
                              ▼
                    ┌──────────────────┐
                    │  WRITER AGENT    │
                    │  Reads JSON →    │
                    │  writes report   │
                    └──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  HUMAN CHECKPOINT │
                    │  "Approve final   │
                    │   report?"        │
                    └─────────┬─────────┘
                              │
                              ▼
                        Final Report
```

---

## Shared Memory Schema

```python
# sessions/{slug}.json

{
    "topic": "Impact of LLMs on software developer jobs",
    "slug": "impact-of-llms-on-software-developer-jobs",
    "status": "researching",  # researching | awaiting_approval | writing | complete
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:05:00Z",
    "research": {
        "academic":  { "status": "complete", "findings": [...], "sources": [...] },
        "industry":  { "status": "running",  "findings": [],    "sources": []    },
        "community": { "status": "failed",   "error": "...",    "findings": []   },
        "economic":  { "status": "complete", "findings": [...], "sources": [...] }
    },
    "checkpoints": [
        {
            "id": 1,
            "type": "review_research",
            "message": "Research complete. Review findings before writing begins.",
            "status": "pending",  # pending | approved | edited
            "user_notes": ""
        }
    ],
    "final_report": null
}
```

---

## Parallel Execution with asyncio

```python
import asyncio
import json
from datetime import datetime

async def run_research_phase(session: dict) -> dict:
    """Run all 4 research agents in parallel."""
    
    roles = ["academic", "industry", "community", "economic"]
    
    async def run_one_agent(role: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {role} agent...")
        try:
            findings = await research_agent.run(
                role=role,
                topic=session["topic"]
            )
            session["research"][role]["status"] = "complete"
            session["research"][role]["findings"] = findings["findings"]
            session["research"][role]["sources"] = findings["sources"]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {role} agent complete ✓")
        except Exception as e:
            session["research"][role]["status"] = "failed"
            session["research"][role]["error"] = str(e)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {role} agent FAILED: {e}")
        finally:
            save_session(session)  # persist after every agent completes

    # This is the key line — all 4 run simultaneously
    await asyncio.gather(*[run_one_agent(role) for role in roles])
    
    return session
```

---

## Human-in-the-Loop Implementation

```python
import json

async def human_checkpoint(session: dict, checkpoint_id: int) -> bool:
    """
    Pause execution and wait for human approval.
    Returns True to continue, False to abort.
    """
    checkpoint = session["checkpoints"][checkpoint_id - 1]
    
    print("\n" + "="*60)
    print(f"CHECKPOINT {checkpoint_id}: {checkpoint['message']}")
    print("="*60)
    
    # Show current state
    print("\nCurrent research status:")
    for role, data in session["research"].items():
        status_icon = "✓" if data["status"] == "complete" else "✗" if data["status"] == "failed" else "..."
        print(f"  {status_icon} {role}: {data['status']}")
    
    print(f"\nSession file: sessions/{session['slug']}.json")
    print("You can edit this file before continuing.\n")
    
    while True:
        choice = input("Type 'yes' to continue, 'edit' to modify the session file, or 'abort' to stop: ").strip().lower()
        
        if choice == "yes":
            checkpoint["status"] = "approved"
            save_session(session)
            return True
        elif choice == "edit":
            input("Edit the JSON file now. Press Enter when done...")
            # Reload from disk in case user edited it
            session = load_session(session["slug"])
            print("Session reloaded from disk.")
        elif choice == "abort":
            print("Aborting. Session saved. Run again to resume.")
            return False
        else:
            print("Invalid input. Type 'yes', 'edit', or 'abort'.")
```

---

## Resumption Logic

```python
async def main():
    topic = input("Research topic: ").strip()
    slug = topic.lower().replace(" ", "-")[:50]
    
    # Check if session already exists
    existing = load_session(slug)
    
    if existing:
        print(f"\nExisting session found. Status: {existing['status']}")
        resume = input("Resume from where you left off? (yes/no): ").strip().lower()
        
        if resume == "yes":
            session = existing
        else:
            session = create_new_session(topic, slug)
    else:
        session = create_new_session(topic, slug)

    # Pick up from wherever we are
    if session["status"] == "researching":
        session = await run_research_phase(session)
        session["status"] = "awaiting_approval"
        save_session(session)

    if session["status"] == "awaiting_approval":
        approved = await human_checkpoint(session, checkpoint_id=1)
        if not approved:
            return
        session["status"] = "writing"
        save_session(session)

    if session["status"] == "writing":
        report = await writer_agent.run(session)
        session["final_report"] = report
        session["status"] = "complete"
        save_session(session)
        print("\n✓ Report complete!")
        print(report)
```

---

## Failure Handling

```python
# After all agents complete, check failure rate
failed = [r for r, d in session["research"].items() if d["status"] == "failed"]
complete = [r for r, d in session["research"].items() if d["status"] == "complete"]

if len(failed) >= 3:
    print(f"Too many agents failed ({len(failed)}/4). Aborting.")
    print("Check your API keys and network connection.")
    return

if len(failed) > 0:
    print(f"Warning: {len(failed)} agent(s) failed: {', '.join(failed)}")
    print("Report will note these gaps.")
    # Continue — Writer handles missing data gracefully
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| OpenRouter | LLM API |
| `asyncio` | Parallel agent execution (built into Python) |
| `httpx` | Async HTTP (works with asyncio natively) |
| `pydantic` | Validate agent outputs |
| `python-dotenv` | Env vars |
| `json` | Shared memory persistence (built into Python) |
| Serper.dev | Search API for research agents |
| Jina Reader | Web page reading |

No new libraries beyond what you've already used.

---

## File Structure

```
autonomous-reporter/
├── main.py                      ← entry point, loads or creates session
├── coordinator.py               ← spawns agents, manages checkpoints
├── agents/
│   ├── __init__.py
│   ├── researcher.py            ← base agent, parameterized by role
│   └── writer.py
├── memory/
│   ├── store.py                 ← load_session / save_session helpers
│   └── sessions/                ← one JSON file per topic (auto-created)
├── checkpoints/
│   └── human.py                 ← pause, prompt, resume logic
├── tools/
│   ├── __init__.py
│   ├── search.py
│   └── read.py
├── prompts/
│   ├── researcher.py            ← template, parameterized by role
│   └── writer.py
├── requirements.txt
└── .env
```

**requirements.txt:**
```
httpx
pydantic
python-dotenv
```

---

## Acceptance Criteria

1. 4 research agents run in parallel — verify with timestamps in console output showing overlapping execution
2. If one agent fails, others continue and final report notes the gap
3. Human checkpoint correctly pauses and waits for input
4. After editing the session JSON by hand, system picks up the edits correctly
5. Kill the process after research completes, restart, confirm it skips to writing phase
6. Final report synthesizes all angles coherently
7. You can explain what `asyncio.gather()` does vs running agents sequentially
8. You can explain why shared mutable state (the JSON file) needs careful handling

---

## Checkpoints

**Checkpoint 1:** Two agents run in parallel. Print their start/end timestamps. Confirm they overlap, not sequence.

**Checkpoint 2:** One agent intentionally raises an Exception. Confirm the other three complete and the failure is recorded in the session JSON.

**Checkpoint 3:** Human checkpoint pauses. Edit the session JSON by hand during the pause. Confirm system loads your changes when you type "done".

**Checkpoint 4:** Ctrl+C after research completes. Restart `main.py`. Confirm it reads from disk and goes directly to the writing phase.

**Checkpoint 5:** Full end-to-end run on a real topic produces a real report.

---

---

# Learning Path Summary

| # | Project | Core Primitive | Key Python concept |
|---|---|---|---|
| 1 | Research Agent | Tool use + Agent loop | `async/await`, `httpx` |
| 2 | Planner + Executor | Multi-agent orchestration | `pydantic`, topological sort |
| 3 | Knowledge Base | Memory + RAG | `chromadb`, embeddings |
| 4 | Code Review Pipeline | Sequential pipelines + critique loops | `pydantic` unions, `ast.parse()` |
| 5 | Autonomous Reporter | Parallel + HITL + persistence | `asyncio.gather()`, JSON persistence |

After Project 5, you will have touched every major architectural primitive in agentic AI — in the same language the entire industry uses. The next layer is framework-level work (LangGraph, CrewAI, AutoGen), but you'll understand exactly what those frameworks are abstracting because you built it yourself first.