@echo off
echo Starting Research Agent Suite...

start cmd /k "cd d:\agentic\research-agent && python server.py"
echo Backend starting on http://localhost:8000...

start cmd /k "cd d:\agentic\research-agent-ui && npm run dev"
echo Frontend starting on http://localhost:3000...

echo Done! Both servers are launching in separate windows.
