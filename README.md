# Rootix AI

Rootix AI is an intelligent root-cause analysis platform for distributed systems. It combines trace + log context with an agent-driven reasoning workflow to help engineers quickly answer questions like:

- Why is this trace slow?
- What failed first?
- Which span/log is the likely root cause?

The system is designed for self-hosted operation and supports a layered architecture from telemetry ingestion to intelligent analysis and feedback-driven improvement.

## Why This Project Is Helpful

- Reduces Mean Time To Resolution (MTTR) by correlating traces and logs automatically.
- Provides guided AI explanations rather than raw telemetry dumps.
- Surfaces evidence-backed references (span IDs, log snippets, function names).
- Captures user feedback to improve analysis quality over time.
- Enables a path to fast, pattern-driven root cause detection before expensive LLM reasoning.

## What This Project Contains

The repository includes:

- Backend services (Python/FastAPI) for trace/log retrieval and chat reasoning orchestration.
- Data access layer for SQLite and MongoDB.
- Agent routing and RCA logic.
- Frontend (Next.js + React) chat interface with trace-aware assistant mode.
- Reasoning and metadata UX components for transparency of AI decisions.

## Run Debugger Agent Locally

Use the following startup sequence to run the full Rootix Debugger Agent stack.

1. Start Jaeger (required for trace storage + query):

```bash
docker compose up -d jaeger
```

2. Start the Python backend API (from repository root):

```bash
python -m src.api.server
```

The backend listens on `http://localhost:8000`.

3. Configure and start the UI (in a second terminal):

```bash
cd ui
cp .env.example .env.local
npm install
npm run dev
```

The UI listens on `http://localhost:3000` and calls the backend via `REST_API_ENDPOINT`.

4. (Optional) Generate sample traces so Explore/Agent has data(For now the SDK is not implemented):

```bash
python sample_app.py
```

5. Open the app at `http://localhost:3000` and use Explore + Agent mode.

### Quick Health Checks

- Jaeger UI: `http://localhost:16686`
- Backend health: `http://localhost:8000/health`
- UI: `http://localhost:3000`

## High-Level Architecture

The architecture follows a layered pipeline:

1. Ingest
- Fetch trace and log telemetry from provider clients.

2. Build Tree
- Convert raw trace + logs into a structured heterogeneous span tree.

3. Intelligence Layer
- Classify spans, suppress noise, locate failures, match patterns, rank causes.

4. Reasoning Layer
- Route to the appropriate agent and generate final response.

5. Persist & Learn
- Store chat history, metadata, reasoning chunks, and intelligence metrics.
- Use user feedback for evaluator loops and model/ranking improvements.

### Backend Layering Pattern

The feedback and chat flow uses a clean layered model:

- Router Layer: HTTP concerns (request parsing, auth, rate limit, errors).
- Driver Layer: business logic orchestration.
- DAO Layer: persistence operations (SQLite/MongoDB).

### Frontend + BFF Pattern

- UI sends requests to Next.js API routes.
- Next.js API routes proxy to Python backend endpoints.
- This keeps frontend auth/session concerns separated from backend service logic.

## End-to-End Chat Flow (How It Works)

1. User asks in Agent mode: "Summarise this Trace".
2. UI posts to BFF endpoint `/api/chat`.
3. BFF forwards request to backend `/v1/explore/post-chat`.
4. Backend validates payload and runs ChatLogic.
5. ChatLogic fetches telemetry, builds context tree, routes to suitable agent.
6. Agent returns structured answer + references + metadata.
7. UI renders answer with:
- Intelligence metadata bar (confidence, fast path, pattern, latency)
- Reasoning section
- Feedback controls (thumbs up/down)
8. Feedback is sent to `/api/feedback`, then persisted in `intelligence_metrics`.
9. Evaluator jobs consume feedback to tune ranking behavior.

## Data Persistence Model

Core stored artifacts include:

- `chat_records`: user and assistant messages
- `chat_metadata`: chat session summary/title
- `reasoning_records`: reasoning chunks per chat
- `chat_routing`: routing decisions
- `intelligence_metrics`: RCA metrics + user feedback

## What Is New (Current Additions)

Recent implementation additions include:

- Feedback API route in UI BFF and backend layered flow.
- Feedback persistence support tied to intelligence metrics.
- Enhanced chat metadata and reasoning route handling.
- Assistant message transparency UI:
- Intelligence metadata bar (confidence/pattern/fast path/time/causes)
- Feedback actions (thumbs up/down with optimistic state)
- Improved reasoning polling behavior to avoid runaway request loops.
- Better compatibility handling in BFF chat payload normalization.

## Future Scope

Temporary note:
- SDK-based production application connectivity to Rootix is not implemented yet.
- This will be implemented in a future release.

1. GitHub API integration so users can create GitHub issues directly from RCA results.
2. AgentOps integration for observability, evaluation, and lifecycle analytics of agents.
3. Implement an eBPF (Extended Berkeley Packet Filter) agent for deeper low-level runtime diagnostics.
4. Implement OpenLLMetry ("Eat Your Own Dogfood") to trace and evaluate LLM/agent behavior end-to-end.

## Recommended Next Milestones

- Harden production deployment profile (auth, retries, circuit breakers, structured errors).
- Expand pattern library coverage for common failure motifs (DB, network, queue, LLM limits).
- Add evaluator dashboards for feedback quality and trend tracking.
- Add integration tests for full UI -> BFF -> backend -> DAO flows.

## Vision

Rootix AI aims to become a feedback-learning reliability copilot: not just explaining incidents, but continuously improving how quickly and accurately root causes are identified in real production systems.
