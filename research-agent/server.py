import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="Research Agent API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    question: str

@app.post("/research")
async def start_research(request: ResearchRequest):
    """Run research and return the full result (non-streaming)."""
    try:
        result = await run_agent(request.question, verbose=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream")
async def stream_research(question: str):
    """
    Experimental SSE stream to show progress.
    Note: We'd need to modify agent.py to yield steps to make this truly useful.
    For now, we'll demonstrate a basic wrapper.
    """
    async def event_generator():
        yield {
            "event": "message",
            "data": json.dumps({"status": "thinking", "message": f"Starting research for: {question}"})
        }
        
        try:
            # We call the agent. In a real production app, we'd modify agent.py 
            # to take a callback for progress updates.
            result = await run_agent(question, verbose=False)
            
            yield {
                "event": "message",
                "data": json.dumps({"status": "complete", "result": result})
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
