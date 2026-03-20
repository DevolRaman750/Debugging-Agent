# Rootix UI — Complete Reference Document

> This file is a self-contained reference for the entire Rootix frontend UI, backend API layer, data models, component tree, and every data flow. It is designed to be given to an LLM as full context for understanding, modifying, or extending Rootix.

---

## TABLE OF CONTENTS

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [Directory Structure](#3-directory-structure)
4. [Pages and Routing](#4-pages-and-routing)
5. [UI Layout — The Explore Page (Main Dashboard)](#5-ui-layout--the-explore-page-main-dashboard)
6. [All Data Models (TypeScript Interfaces)](#6-all-data-models-typescript-interfaces)
7. [All API Endpoints (17 Total)](#7-all-api-endpoints-17-total)
8. [Component Tree and Ownership](#8-component-tree-and-ownership)
9. [Flow 1 — Trace Fetching](#flow-1--trace-fetching)
10. [Flow 2 — Log Fetching](#flow-2--log-fetching)
11. [Flow 3 — AI Chat (Core Pipeline)](#flow-3--ai-chat-core-pipeline)
12. [Flow 4 — Chat History and Metadata](#flow-4--chat-history-and-metadata)
13. [Flow 5 — Reasoning Steps](#flow-5--reasoning-steps)
14. [Flow 6 — GitHub Action Confirmation](#flow-6--github-action-confirmation)
15. [Flow 7 — Source Code Context](#flow-7--source-code-context)
16. [Flow 8 — Provider Configuration](#flow-8--provider-configuration)
17. [Flow 9 — Integration Tokens](#flow-9--integration-tokens)
18. [Flow 10 — OTLP Ingestion](#flow-10--otlp-ingestion)
19. [Authentication](#19-authentication)
20. [Environment Variables](#20-environment-variables)

---

## 1. System Architecture Overview

Rootix is a 2-layer proxy architecture:

```
Browser (Next.js App)
    │
    ▼
Next.js API Routes (ui/src/app/api/*)          ← BFF (Backend-For-Frontend)
    │                        │
    │ (9 endpoints)          │ (6 endpoints)
    ▼                        ▼
Python FastAPI Backend       MongoDB (Direct)
(REST_API_ENDPOINT)
    │
    ├──► Jaeger / AWS X-Ray / Tencent   (trace + log sources)
    ├──► LLM (GPT-4o / GPT-4.1 / GPT-5) (AI reasoning)
    ├──► GitHub API                      (issue/PR creation, code fetch)
    ├──► MongoDB                         (chat records, metadata, reasoning)
    └──► SQLite (local mode)             (alternative persistence)
```

**Key design point**: The Next.js API routes act as a BFF proxy. Some routes forward requests to the Python FastAPI backend (telemetry, chat, code). Other routes go directly to MongoDB (provider config, integration tokens). The browser never calls the Python backend or MongoDB directly.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Frontend framework | Next.js 14+ (App Router) |
| UI language | TypeScript + React |
| Styling | TailwindCSS + custom CSS |
| Component library | shadcn/ui (Radix primitives) |
| Auth | Clerk (sign-in, sign-up, session tokens) |
| Icons | lucide-react, react-icons (Ri, Go, Fa, Si, Io, Md) |
| Backend framework | Python FastAPI |
| Backend language | Python 3.x |
| Primary database | MongoDB (Mongoose on frontend, motor on backend) |
| Alternative database | SQLite (aiosqlite, local mode) |
| Trace sources | Jaeger, AWS X-Ray, Tencent Cloud |
| LLM providers | OpenAI (GPT-4o, GPT-4.1, GPT-5), Groq |
| OTLP ingestion | OpenTelemetry Collector → protobuf |
| Billing | Autumn (external service) |
| Package manager | bun / npm |

---

## 3. Directory Structure

```
rootix-main/
├── ui/                            ← Next.js frontend
│   └── src/
│       ├── app/
│       │   ├── api/               ← Next.js API routes (BFF proxy layer)
│       │   │   ├── chat/          ← POST: send message to AI agent
│       │   │   ├── list_trace/    ← GET: list traces from Jaeger/AWS
│       │   │   ├── get_trace_log/ ← GET: fetch logs by trace ID
│       │   │   ├── get_chat_history/         ← GET: fetch chat messages
│       │   │   ├── get_chat_metadata/        ← GET: fetch chat title/metadata
│       │   │   ├── get_chat_metadata_history/ ← GET: list past chats for trace
│       │   │   ├── confirm-github-action/    ← POST: confirm/reject GitHub action
│       │   │   ├── get_line_context_content/  ← GET: fetch source code lines
│       │   │   ├── provider-config/          ← GET/POST/DELETE: provider settings
│       │   │   ├── get_connect/    ← GET: fetch integration tokens
│       │   │   ├── post_connect/   ← POST: save integration tokens
│       │   │   ├── delete_connect/ ← DELETE: remove integration tokens
│       │   │   └── autumn/         ← ALL: proxy to Autumn billing
│       │   ├── explore/           ← Main dashboard page
│       │   ├── integrate/         ← SDK integration page
│       │   ├── settings/          ← Provider settings page
│       │   ├── pricing/           ← Pricing page
│       │   ├── sign-in/           ← Clerk sign-in
│       │   ├── sign-up/           ← Clerk sign-up
│       │   ├── login/             ← Login redirect
│       │   ├── layout.tsx         ← Root layout
│       │   ├── layout-client.tsx  ← Client layout wrapper
│       │   ├── authenticated-layout.tsx ← Auth-gated layout
│       │   ├── page.tsx           ← Root page (redirects to /explore)
│       │   └── globals.css        ← Global styles
│       ├── components/
│       │   ├── explore/           ← Trace list components
│       │   │   ├── Trace.tsx      ← Main trace list (1117 lines)
│       │   │   ├── SearchBar.tsx  ← Multi-criterion search
│       │   │   ├── TimeButton.tsx ← Time range selector
│       │   │   ├── CustomTimeRangeDialog.tsx ← Custom time picker
│       │   │   ├── ExploreHeader.tsx ← Header with search + time
│       │   │   ├── RefreshButton.tsx ← Manual refresh
│       │   │   └── span/
│       │   │       └── Span.tsx   ← Nested span tree renderer
│       │   ├── right-panel/       ← Right side of dashboard
│       │   │   ├── ModeToggle.tsx ← Log/Trace/Agent view switcher
│       │   │   ├── RightPanelSwitch.tsx ← Routes to log or trace view
│       │   │   ├── agent/         ← AI chat panel
│       │   │   │   ├── Agent.tsx  ← Chat orchestrator (733 lines)
│       │   │   │   ├── TopBar.tsx ← Chat tabs + history dropdown
│       │   │   │   ├── ChatMessage.tsx ← Message renderer with markdown
│       │   │   │   ├── MessageInput.tsx ← Input with mode/model selectors
│       │   │   │   └── chat-reasoning.tsx ← Reasoning steps accordion
│       │   │   ├── log/           ← Log view components
│       │   │   │   ├── LogPanelSwitch.tsx
│       │   │   │   ├── LogDetail.tsx ← Log entry display
│       │   │   │   ├── LogMetadataFilter.tsx
│       │   │   │   ├── LogSearchInput.tsx
│       │   │   │   └── ShowCodeToggle.tsx ← Source code viewer
│       │   │   └── trace/         ← Trace waterfall view
│       │   │       ├── TracePanelSwitch.tsx
│       │   │       ├── TraceDetail.tsx
│       │   │       └── TraceDepthChart.tsx
│       │   ├── side-bar/          ← Navigation sidebar
│       │   ├── settings/          ← Settings page components
│       │   │   ├── TraceProviderTabContent.tsx
│       │   │   ├── LogProviderTabContent.tsx
│       │   │   └── ... (other settings tabs)
│       │   ├── integrate/         ← Integration page components
│       │   │   ├── Item.tsx       ← Token input/save/delete
│       │   │   └── RightPanel.tsx ← Integration docs
│       │   ├── auth/              ← Auth components
│       │   ├── plot/              ← Chart/graph components
│       │   ├── resizable/         ← Resizable panel layouts
│       │   └── ui/               ← shadcn/ui primitives (48 components)
│       ├── models/               ← TypeScript data models
│       │   ├── chat.ts           ← Chat request/response interfaces
│       │   ├── trace.ts          ← Trace/Span interfaces
│       │   ├── log.ts            ← LogEntry/TraceLog interfaces
│       │   ├── code.ts           ← CodeResponse interface
│       │   ├── integrate.ts      ← ResourceType enum
│       │   ├── provider.ts       ← Provider config Mongoose schemas
│       │   └── token.ts          ← Token Mongoose schemas
│       ├── constants/            ← App constants
│       │   ├── model.ts          ← LLM model names, providers
│       │   ├── colors.ts         ← Percentile color mapping
│       │   └── animations.ts     ← CSS animation constants
│       ├── hooks/                ← React hooks
│       ├── lib/                  ← Utility libraries
│       │   ├── clerk-auth.ts     ← Clerk auth helpers
│       │   ├── server-auth-headers.ts ← Backend auth header builder
│       │   ├── mongodb.ts        ← MongoDB connection utility
│       │   └── utils.ts          ← General utilities
│       ├── utils/                ← Client utilities
│       │   ├── provider.ts       ← Provider param builder
│       │   └── uuid.ts           ← UUID generator
│       ├── providers/            ← React context providers
│       ├── types/                ← Additional type definitions
│       └── middleware.ts         ← Clerk auth middleware
├── rest/                         ← Python FastAPI backend
│   └── routers/
│       ├── chat.py               ← ChatRouterClass (6 endpoints)
│       ├── telemetry.py          ← TelemetryRouter (2 endpoints)
│       ├── internal.py           ← InternalRouter (1 endpoint)
│       └── verify.py             ← Health check
├── src/                          ← Python business logic
│   ├── agents/                   ← AI agent implementations
│   ├── dao/                      ← Database access objects
│   ├── routing/                  ← Response builder + types
│   └── intel/                    ← Evaluation loop + config
└── kan.md                        ← Pipeline architecture doc
```

---

## 4. Pages and Routing

| Route | Page | Auth Required | Description |
|---|---|---|---|
| `/` | `page.tsx` | Yes | Redirects to `/explore` |
| `/explore` | `explore/page.tsx` | Yes | Main dashboard: trace list + log/trace view + agent chat |
| `/integrate` | `integrate/page.tsx` | Yes | SDK integration: manage API tokens (Rootix, GitHub, etc.) |
| `/settings` | `settings/page.tsx` | Yes | Provider config: Jaeger, AWS, Tencent connection settings |
| `/pricing` | `pricing/page.tsx` | No | Pricing plans (Autumn billing integration) |
| `/sign-in` | `sign-in/page.tsx` | No | Clerk sign-in page |
| `/sign-up` | `sign-up/page.tsx` | No | Clerk sign-up page |
| `/login` | `login/page.tsx` | No | Login redirect |

Auth is enforced by Clerk middleware in `middleware.ts`. The protected route pattern is `"/explore(.*)"`, `"/integrate(.*)"`, `"/settings(.*)"`. API routes under `/api/autumn(.*)` are excluded from middleware.

---

## 5. UI Layout — The Explore Page (Main Dashboard)

The explore page is a resizable 3-panel layout:

```
┌──────────┬─────────────────────────────┬──────────────────────────────────────┐
│          │                             │  [📄 Log] [🔭 Trace] [🤖 Agent]     │
│ SIDEBAR  │  LEFT PANEL                 │                                      │
│          │  ┌───────────────────────┐   │  RIGHT PANEL (top)                   │
│ Explore  │  │ 🔍 SearchBar          │  │  ┌──────────────────────────────┐    │
│ Integrate│  │ ⏱ TimeButton          │  │  │ LOG VIEW or TRACE VIEW       │    │
│ Settings │  │ 🔄 RefreshButton      │  │  │ (LogPanelSwitch or           │    │
│ Pricing  │  ├───────────────────────┤  │  │  TracePanelSwitch)           │    │
│          │  │ ☐ 🐍 abc12... P95     │  │  │                              │    │
│          │  │   svc-checkout 120ms ❌│  │  │ Shows logs with level colors │    │
│          │  │   └ span tree...      │  │  │ or trace waterfall with bars │    │
│          │  │ ☐ TS def45... P50     │  │  └──────────────────────────────┘    │
│          │  │   svc-auth 40ms  ✅   │  │                                      │
│          │  │ ☐ ☕ ghi78... P75     │  │  RIGHT PANEL (bottom) — Agent Chat   │
│          │  │   svc-payment 85ms ⚠️ │  │  ┌──────────────────────────────┐    │
│          │  │   └ expanded spans... │  │  │ [Tab1 ×] [Tab2 ×]    [+] [⏱]│    │
│          │  │                       │  │  │ ┌────────────────────────┐   │    │
│          │  │ [Load More]           │  │  │ │ 🤖 "N+1 pattern [1]"  │   │    │
│          │  └───────────────────────┘  │  │ │    "in get_cart [2]"   │   │    │
│          │                             │  │ │ ▸ Reasoning (collapse) │   │    │
│          │                             │  │ │ 👤 "Create a fix?"     │   │    │
│          │                             │  │ └────────────────────────┘   │    │
│          │                             │  │ trace: abc12.. | 3 spans     │    │
│          │                             │  │ [Agent▼] [GPT-4o▼]    [→]   │    │
│          │                             │  └──────────────────────────────┘    │
└──────────┴─────────────────────────────┴──────────────────────────────────────┘
```

### Left Panel — Trace List (`Trace.tsx`, 1117 lines)

**Header**: `ExploreHeader.tsx` + `SearchBar.tsx` + `TimeButton.tsx`
- SearchBar supports multi-criterion filtering: category (service_name, span_name, status, etc.) + operation (equals, contains, not_equals) + value
- TimeButton: presets (5m, 15m, 30m, 1h, 3h, 6h, 12h, 24h) + custom absolute/relative ranges
- Timezone toggle: UTC vs local (`CustomTimeRangeDialog.tsx`)

**Trace Rows**: Each trace is a 43px row showing:
- Checkbox (multi-select with shift-click range selection)
- SDK language icon: Python 🐍, TypeScript, JavaScript, Java ☕
- Trace ID badge: first 8 chars with tooltip for full ID
- Percentile tag: P50 (green), P75 (yellow), P90 (orange), P99 (red)
- Service name(s): truncated to 25 chars
- Duration: formatted as µs/ms/s/m
- Error badges: warning ⚠️ count, error ❌ count
- Share button: copies URL with `?trace_id=` parameter

**Behavior**:
- Auto-selects first trace on load
- `?trace_id=` in URL auto-selects that trace
- Expanding a trace shows nested span tree (`Span.tsx`)
- Pagination: "Load More" button with `pagination_token`
- On trace select: `onTraceSelect(traceIds)` → updates right panel
- On span select: `onSpanSelect(spanIds)` → updates right panel

### Right Panel — View Switcher (`ModeToggle.tsx`)

Three toggle buttons in top-right:
- **Log view** (`FileCode2` icon) → `LogPanelSwitch` → `LogDetail.tsx`
- **Trace view** (`Telescope` icon) → `TracePanelSwitch` → `TraceDetail.tsx`
- **Agent toggle** (`RiRobot2Line` icon) → opens/closes chat panel

`RightPanelSwitch.tsx` receives `traceIds`, `spanIds`, `allTraces`, and `viewType` props. It loads spans from selected traces and routes to either `LogPanelSwitch` (viewType="log") or `TracePanelSwitch` (viewType="trace").

### Agent Chat Panel (`Agent.tsx`, 733 lines)

**Structure**:
1. **TopBar** — chat tabs + history dropdown
2. **ChatMessage** — message list with markdown rendering
3. **MessageInput** — input textarea with mode/model selectors

Detailed descriptions of each are in the flow sections below.

---

## 6. All Data Models (TypeScript Interfaces)

### Trace Models (`ui/src/models/trace.ts`)

```typescript
interface Span {
  id: string;
  name: string;
  start_time: number;         // Unix timestamp in seconds
  end_time: number;
  duration: number;           // In seconds
  num_debug_logs?: number;
  num_info_logs?: number;
  num_warning_logs?: number;
  num_error_logs?: number;
  num_critical_logs?: number;
  spans?: Span[];             // Nested child spans (recursive)
  telemetry_sdk_language?: string;
}

interface Trace {
  id: string;
  service_name?: string;
  service_environment?: string;
  num_debug_logs?: number;
  num_info_logs?: number;
  num_warning_logs?: number;
  num_error_logs?: number;
  num_critical_logs?: number;
  duration: number;
  start_time: number;
  end_time: number;
  percentile: string;         // "P50", "P75", "P90", "P95", "P99"
  spans: Span[];
  telemetry_sdk_language: string[];  // ["python", "ts", "js", "java"]
}

interface TraceResponse {
  success: boolean;
  data: Trace[];
  error?: string;
  next_pagination_token?: string;
  has_more?: boolean;
}
```

### Log Models (`ui/src/models/log.ts`)

```typescript
enum LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL }

interface LogEntry {
  time: number;
  function_name: string;
  level: LogLevel;
  message: string;
  file_name: string;
  line_number: number;
  trace_id?: string;
  span_id?: string;
  git_url?: string;           // GitHub URL to source code
  commit_id?: string;
  line?: string;              // Source code line content
  lines_above?: string[];     // Context lines above
  lines_below?: string[];     // Context lines below
}

interface SpanLog { [spanId: string]: LogEntry[] }
interface TraceLog { [traceId: string]: SpanLog[] }
```

### Chat Models (`ui/src/models/chat.ts`)

```typescript
type MessageType = "assistant" | "user" | "github" | "statistics";
type ActionType = "github_get_file" | "agent_chat" | "pending_confirmation"
               | "github_create_issue" | "github_create_pr";
type ActionStatus = "pending" | "success" | "failed" | "cancelled" | "awaiting_confirmation";
type Provider = "openai" | "custom";

interface Reference {
  number: number;              // [1], [2], etc. in message text
  span_id?: string;
  span_function_name?: string;
  line_number?: number;
  log_message?: string;
}

interface ChatRequest {
  time: number;
  message: string;
  message_type: MessageType;
  trace_id: string;
  span_ids: string[];
  start_time: number;
  end_time: number;
  model: ChatModel;            // "gpt-4o" | "gpt-4.1" | "gpt-5" | "auto"
  mode: ChatMode;              // "agent" | "chat"
  chat_id: string;
  trace_provider: string;      // "jaeger" | "aws" | "tencent"
  log_provider: string;
  trace_region?: string;
  log_region?: string;
  provider: Provider;
}

interface ChatbotResponse {
  time: number;
  message: string;             // Markdown with [1] [2] references
  reference: Reference[];
  message_type: MessageType;
  chat_id: string;
  action_type?: ActionType;
  status?: ActionStatus;
}

interface ChatResponse {
  success: boolean;
  data: ChatbotResponse | null;
  error?: string;
}

interface ChatMetadata {
  chat_id: string;
  timestamp: number;
  chat_title: string;
  trace_id: string;
  user_id?: string;
}

interface ChatMetadataHistory {
  history: ChatMetadata[];
  hasMore?: boolean;
}

interface ChatHistoryResponse {
  history: ChatbotResponse[];
}

// GitHub action confirmation
interface ConfirmActionRequest {
  chat_id: string;
  message_timestamp: number;
  confirmed: boolean;
}

interface ConfirmActionResponse {
  success: boolean;
  message: string;
  data?: Record<string, any>;
}

// Reasoning records (MongoDB collection: reasoning_streams)
interface IReasoningRecord {
  chat_id: string;
  chunk_id: number;
  content: string;
  status: string;              // "in_progress" | "complete"
  timestamp: Date;
  trace_id?: string;
}
```

### Code Models (`ui/src/models/code.ts`)

```typescript
interface CodeResponse {
  line: string | null;          // The source code line at the error
  lines_above: string[] | null; // Context lines above
  lines_below: string[] | null; // Context lines below
  error_message: string | null;
}
```

### Integration Models (`ui/src/models/integrate.ts`)

```typescript
enum ResourceType { GITHUB, NOTION, SLACK, OPENAI, ROOTIX }

interface TokenResource {
  token?: string | null;
  resourceType: ResourceType;
}
```

---

## 7. All API Endpoints (17 Total)

### Endpoints That Proxy to Python Backend (9)

These go through Next.js API routes → Python FastAPI backend (`REST_API_ENDPOINT`):

| # | Frontend Route | HTTP | Python Backend URL | Purpose |
|---|---|---|---|---|
| 1 | `/api/list_trace` | GET | `/v1/explore/list-traces` | List traces with filters |
| 2 | `/api/get_trace_log` | GET | `/v1/explore/get-logs-by-trace-id` | Fetch logs for a trace |
| 3 | `/api/chat` | POST | `/v1/explore/post-chat` | Send message to AI agent |
| 4 | `/api/get_chat_history` | GET | `/v1/explore/get-chat-history` | Fetch chat messages |
| 5 | `/api/get_chat_metadata` | GET | `/v1/explore/get-chat-metadata` | Fetch chat title |
| 6 | `/api/get_chat_metadata_history` | GET | `/v1/explore/get-chat-metadata-history` | List past chats for a trace |
| 7 | `/api/chat/[id]/reasoning` | GET | (queries MongoDB reasoning_streams) | Fetch reasoning steps |
| 8 | `/api/confirm-github-action` | POST | `/v1/explore/confirm-github-action` | Confirm/reject GitHub action |
| 9 | `/api/get_line_context_content` | GET | `/v1/explore/get-line-context-content` | Fetch source code lines |

### Endpoints That Go Directly to MongoDB (6)

These **skip the Python backend entirely**. The Next.js API route connects to MongoDB via Mongoose:

| # | Frontend Route | HTTP | MongoDB Collection | Purpose |
|---|---|---|---|---|
| 10 | `/api/provider-config` | GET | `trace_provider_configs` + `log_provider_configs` | Load provider settings |
| 11 | `/api/provider-config` | POST | same | Save provider settings |
| 12 | `/api/provider-config` | DELETE | same | Delete provider settings |
| 13 | `/api/get_connect` | GET | `rootix_tokens` or `connection_tokens` | Fetch integration token |
| 14 | `/api/post_connect` | POST | same | Save integration token |
| 15 | `/api/delete_connect` | DELETE | same | Delete integration token |

### Other Endpoints (2)

| # | Route | HTTP | Destination | Purpose |
|---|---|---|---|---|
| 16 | `/api/autumn/[...all]` | ALL | Autumn billing API | Proxy billing/subscription requests |
| 17 | (no frontend) | POST `/v1/traces` | Python InternalRouter | Receive OTLP traces from OTel Collector |

### Python Backend Routers

The Python FastAPI backend at `REST_API_ENDPOINT` has 3 router classes:

**TelemetryRouter** (`rest/routers/telemetry.py`) — prefix: `/v1/explore/`
- `GET /list-traces` — queries Jaeger/AWS/Tencent for traces
- `GET /get-logs-by-trace-id` — queries Jaeger/AWS/Tencent for logs

**ChatRouterClass** (`rest/routers/chat.py`) — prefix: `/v1/explore/`
- `POST /post-chat` — runs the AI pipeline: Ingest → Tree → Intel → Route → LLM → Validate → Persist → Respond
- `GET /get-chat-history` — reads chat_records from MongoDB
- `GET /get-chat-metadata` — reads chat_metadata from MongoDB
- `GET /get-chat-metadata-history` — lists chat_metadata by trace_id
- `GET /get-line-context-content` — fetches file from GitHub API
- `POST /confirm-github-action` — executes pending GitHub action

**InternalRouter** (`rest/routers/internal.py`) — no prefix
- `POST /v1/traces` — receives OTLP protobuf/JSON traces, counts usage

---

## 8. Component Tree and Ownership

```
Explore Page
├── ExploreHeader
│   ├── SearchBar          → builds search criteria [{category, value, operation}, ...]
│   ├── TimeButton         → selects time range (preset or custom)
│   └── RefreshButton      → triggers re-fetch
├── Trace (LEFT PANEL)     → calls /api/list_trace
│   ├── Span (nested)      → rendered inside expanded traces
│   └── Share Dialog       → copies URL with trace_id param
├── ModeToggle             → switches viewType: "log" | "trace", toggles agent
├── RightPanelSwitch       → routes to LogPanelSwitch or TracePanelSwitch
│   ├── LogPanelSwitch
│   │   ├── LogDetail      → calls /api/get_trace_log
│   │   ├── LogSearchInput
│   │   ├── LogMetadataFilter
│   │   └── ShowCodeToggle → calls /api/get_line_context_content
│   └── TracePanelSwitch
│       ├── TraceDetail
│       └── TraceDepthChart
└── Agent (CHAT PANEL)     → calls /api/chat, /api/get_chat_history
    ├── TopBar             → calls /api/get_chat_metadata, /api/get_chat_metadata_history
    │   ├── Chat Tabs (multi-tab)
    │   ├── New Chat button (+)
    │   └── History dropdown (⏱)
    ├── ChatMessage         → renders messages with markdown + references
    │   ├── renderMarkdown() → handles [1] [2] references, code blocks, links, etc.
    │   ├── HoverCard       → reference popup (span_id, function, log, line_number)
    │   ├── ChatReasoning   → calls /api/chat/[id]/reasoning
    │   └── Confirm buttons → "Yes, create it" / "No, cancel" for GitHub actions
    └── MessageInput
        ├── PromptInput     → auto-resizing textarea (1-5 rows)
        ├── Mode selector   → "Agent" (full pipeline) / "Chat" (fast RCA)
        ├── Model selector  → GPT-4o / GPT-4.1 / GPT-5 / Auto
        ├── Context badges  → shows selected trace_id and span count
        └── Send button     → disabled until trace selected + text entered
```

---

## FLOW 1 — Trace Fetching

**Components**: `Trace.tsx` → `/api/list_trace` → Python `TelemetryRouter.list_traces` → Jaeger/AWS/Tencent

**Trigger**: Page load, time range change, search criteria change, or refresh button click.

**Step-by-step**:
1. `Trace.tsx` reads `selectedTimeRange`, `timezone`, `searchCriteria` from page-level state
2. Constructs URL: `/api/list_trace?startTime=...&endTime=...&categories=...&values=...&operations=...&trace_provider=...&log_provider=...`
3. Provider params (`trace_provider`, `log_provider`, `trace_region`, `log_region`) are read from localStorage via `buildProviderParams()`
4. If `?trace_id=` exists in browser URL, it's appended to the API call
5. Next.js route (`list_trace/route.ts`) forwards to `REST_API_ENDPOINT/v1/explore/list-traces` with auth headers
6. Python backend queries the appropriate trace source (Jaeger GRPC, AWS X-Ray API, or Tencent API)
7. Response: `{ traces: Trace[], next_pagination_token?, has_more? }`
8. `Trace.tsx` renders trace rows, auto-selects first trace
9. On "Load More" click: same request with `pagination_token` appended, traces are appended (deduped)

**Query params for `/api/list_trace`**:
- `startTime` (ISO string) — required
- `endTime` (ISO string) — required
- `categories[]` — filter categories (e.g., "service_name", "span_name", "status_code")
- `values[]` — filter values (e.g., "checkout-service", "500")
- `operations[]` — filter operations (e.g., "equals", "contains", "not_equals")
- `trace_provider` — "jaeger" | "aws" | "tencent" (default: "aws")
- `log_provider` — same options
- `trace_region` — AWS region (optional for Jaeger)
- `log_region` — same
- `trace_id` — direct lookup by trace ID (optional)
- `pagination_token` — for next page (optional)

**Timeout**: 60 seconds

---

## FLOW 2 — Log Fetching

**Components**: `LogDetail.tsx` → `/api/get_trace_log` → Python `TelemetryRouter.get_logs_by_trace_id` → Jaeger/AWS/Tencent

**Trigger**: When a trace is selected in the left panel and the right panel is in "log" view mode.

**Step-by-step**:
1. `LogDetail.tsx` receives `traceId`, `spanIds`, `segments` (spans) from `RightPanelSwitch`
2. Constructs URL: `/api/get_trace_log?traceId=...&start_time=...&end_time=...&trace_provider=...&log_provider=...`
3. Next.js route forwards to `REST_API_ENDPOINT/v1/explore/get-logs-by-trace-id`
4. Python backend fetches logs from the configured log source
5. Response: `{ logs: { logs: LogEntry[] } }` — transformed to `TraceLog` format
6. `LogDetail.tsx` renders logs grouped by span, with level-based coloring:
   - CRITICAL → dark red (#7f1d1d)
   - ERROR → red (#dc2626)
   - WARNING → orange (#fb923c)
   - INFO → gray (#64748b)
   - DEBUG → purple (#a855f7)
   - TRACE → indigo (#6366f1)
7. Each log entry shows: timestamp, function_name, level, message, file_name:line_number
8. If `git_url` is present, `ShowCodeToggle.tsx` appears (see Flow 7)

**Timeout**: 90 seconds

---

## FLOW 3 — AI Chat (Core Pipeline)

**Components**: `Agent.tsx` → `MessageInput.tsx` → `/api/chat` → Python `ChatRouterClass.post_chat` → KAN Pipeline → LLM → Response

**Trigger**: User types a message and presses Enter or clicks Send.

### Step-by-step:

1. **Input validation**: Message must be non-empty, a trace must be selected, multiple traces disable the agent, and no ongoing request can be in progress

2. **Chat ID generation**: If this is the first message (new chat), `generateUuidHex()` creates a chat ID. The "New Chat" tab is updated with this ID

3. **User message added to UI**: The message appears immediately in `ChatMessage.tsx` (optimistic update)

4. **API request constructed**:
```json
POST /api/chat
{
  "time": 1708012345000,
  "message": "Why is checkout slow?",
  "messageType": "user",
  "trace_id": "32e4d40f...",
  "span_ids": ["abc123", "def456"],
  "start_time": 1708000000000,
  "end_time": 1708012345000,
  "model": "gpt-4o",
  "mode": "agent",
  "chat_id": "chat_xyz123",
  "trace_provider": "jaeger",
  "log_provider": "jaeger",
  "trace_region": null,
  "log_region": null,
  "provider": "openai"
}
```

5. **Polling starts**: While waiting for the response, `Agent.tsx` polls `GET /api/get_chat_history?chat_id=...` every 1 second to check for GitHub/statistics messages that the backend may have inserted during processing

6. **Python backend runs the KAN pipeline**:
   - Stage 1 (Ingest): Fetch trace + logs from Jaeger/AWS
   - Stage 2 (Tree Builder): Compress trace tree
   - Stage 3 (Intelligence): Pattern matching (N+1, slow_query, retry_storm, etc.)
   - Stage 4 (Router): Fast path (known pattern) or full path (send to LLM)
   - Stage 5 (LLM): Send compressed context + question to GPT-4o
   - Stage 6 (Validation): Verify references point to real spans
   - Stage 7 (Persistence): Store to chat_records + intelligence_metrics
   - Stage 8 (Response Builder): Assemble ChatbotResponse with metadata

7. **Response received**:
```json
{
  "success": true,
  "data": {
    "time": 1708012350000,
    "message": "The checkout endpoint is slow due to an N+1 query pattern in `get_user_cart` [1]. The function makes 47 individual SELECT queries [2].",
    "reference": [
      {"number": 1, "span_id": "abc123", "span_function_name": "get_user_cart", "line_number": 84},
      {"number": 2, "span_id": "def456", "log_message": "SELECT * FROM products WHERE id = ?"}
    ],
    "message_type": "assistant",
    "chat_id": "chat_xyz123",
    "action_type": "agent_chat",
    "status": "success"
  }
}
```

8. **Message rendered**: `ChatMessage.tsx` calls `renderMarkdown()` which:
   - Detects `[1]` → creates a `HoverCard` showing span_id, function_name, line_number, log_message
   - Clicking span_id in HoverCard → calls `onSpanSelect(span_id)` + `onViewTypeChange("log")` → user jumps to that span in log view
   - Detects code blocks → renders with syntax highlighting + copy button
   - Detects GitHub URLs → renders as clickable PR/issue links
   - Detects log levels (ERROR, WARNING) → colored text

9. **Reasoning block**: After every user message, `ChatReasoning` appears as a collapsible accordion showing the agent's thinking steps (see Flow 5)

10. **TopBar refresh**: `topBarRef.current.refreshMetadata()` updates the tab title via animated character-by-character transition

11. **Polling stops**: When the response arrives or an error occurs

### Special: GitHub action flow
If the backend returns `action_type: "pending_confirmation"` and `status: "awaiting_confirmation"`, the message renders with two buttons:
- **"Yes, create it"** (green) → sends "yes" as a follow-up message
- **"No, cancel"** (red outline) → sends "no" as a follow-up message

### Modes:
- **Agent mode**: Full pipeline with GitHub integration, multi-step reasoning, statistics
- **Chat mode**: Fast single-RCA agent, no GitHub actions, faster response

---

## FLOW 4 — Chat History and Metadata

**Components**: `TopBar.tsx` → `/api/get_chat_metadata_history`, `/api/get_chat_metadata`, `/api/get_chat_history`

### History Dropdown (⏱ button):

1. User clicks the clock icon → dropdown opens → `fetchChatHistory()` fires
2. `GET /api/get_chat_metadata_history?trace_id=...&limit=5&skip=0`
3. Response: `{ history: [{chat_id, chat_title, timestamp}, ...], hasMore: boolean }`
4. Items grouped into: Today, Yesterday, Last Week, Last Month, Older
5. Already-open chats show a checkmark ✓
6. Clicking an item:
   - If already in a tab → switches to that tab
   - If not → calls `handleHistoryItemsSelect` → fetches full chat history → creates new tab

### Loading a Chat from History:

1. `GET /api/get_chat_history?chat_id=...` → returns all `ChatbotResponse[]` for that chat
2. Messages are sorted by timestamp, deduplicated by content+role+timestamp
3. `GET /api/get_chat_metadata?chat_id=...` → returns `chat_title`
4. New tab created with title and messages

### Tab Title Animation:

When chat metadata is fetched ('TopBar.tsx'), the title animates character-by-character:
- If the tab showed a truncated title "Why is check...", the animation starts from "Why is check" and types out the remaining characters
- A blinking cursor `|` appears during animation
- Animation speed: 20ms per character

### Multi-Tab System:

- Multiple chat tabs can be open simultaneously
- Each tab has: `chatId | null` (null = new chat), `title`, `messages[]`, `tempId?`
- Only one "New Chat" tab allowed at a time
- Closing the last tab resets to a fresh "New Chat" tab
- Closing active tab switches to the first remaining tab

---

## FLOW 5 — Reasoning Steps

**Components**: `chat-reasoning.tsx` → `/api/chat/[chatId]/reasoning` (queries MongoDB `reasoning_streams`)

**Trigger**: After every user message, a `ChatReasoning` component appears.

1. `ChatReasoning` receives `chatId`, `isLoading`, `userMessageTimestamp`, `nextMessageTimestamp`
2. Calls `GET /api/chat/{chatId}/reasoning?t=...` (timestamp for cache busting)
3. Backend queries MongoDB `reasoning_streams` collection for records with matching `chat_id`
4. Records filtered by timestamp window: after user message, before next assistant message
5. Each record has: `chunk_id`, `content`, `status` ("in_progress" or "complete")
6. Displayed as a collapsible `Reasoning` accordion:
   - `ReasoningTrigger` — "Thinking..." label
   - `ReasoningContent` — shows thinking steps
7. While `isLoading=true`, shows streaming animation
8. Defaults to open (collapsed after interaction)

---

## FLOW 6 — GitHub Action Confirmation

**Components**: `ChatMessage.tsx` → `Agent.tsx` → `/api/chat`

**Trigger**: When the AI agent proposes a GitHub action (create issue, create PR).

1. Backend returns a message with `action_type: "pending_confirmation"` and `status: "awaiting_confirmation"`
2. `ChatMessage.tsx` detects this and renders two buttons:
   - **"Yes, create it"** → resolves to `onSendMessage("yes")`
   - **"No, cancel"** → resolves to `onSendMessage("no")`
3. `Agent.tsx` receives this, sets `inputMessage` to "yes"/"no", and programmatically submits the form
4. A new `POST /api/chat` call is made with the confirmation message
5. Backend executes or cancels the GitHub action
6. POST /api/confirm-github-action is called internally by the backend
7. Backend calls GitHub API (create issue/PR)
8. Response includes the GitHub URL: `"PR created: https://github.com/owner/repo/pull/42"`
9. `ChatMessage.tsx` renders the PR URL as a clickable link: `PR #42`

---

## FLOW 7 — Source Code Context

**Components**: `ShowCodeToggle.tsx` → `/api/get_line_context_content` → Python → GitHub API

**Trigger**: A log entry has a `git_url` field (e.g., `https://github.com/owner/repo/blob/main/file.py#L42`), and the user toggles "Show Code".

1. `ShowCodeToggle.tsx` detects log entries with `git_url` present
2. On toggle: `GET /api/get_line_context_content?url=...` (URL-encoded GitHub URL)
3. Next.js route forwards to `REST_API_ENDPOINT/v1/explore/get-line-context-content`
4. Python backend parses the GitHub URL, fetches the file via GitHub API (requires user's GitHub token from `connection_tokens`)
5. Response:
```json
{
  "line": "        db.query(f\"SELECT * FROM products WHERE id = {item.id}\")",
  "lines_above": ["    for item in cart.items:", "        # fetch each product"],
  "lines_below": ["    return total", ""],
  "error_message": null
}
```
6. `ShowCodeToggle.tsx` renders code with the error line highlighted

---

## FLOW 8 — Provider Configuration

**Components**: `TraceProviderTabContent.tsx`, `LogProviderTabContent.tsx` → `/api/provider-config` → Direct MongoDB

**THIS FLOW SKIPS THE PYTHON BACKEND.** The Next.js API route connects directly to MongoDB via Mongoose.

### Supported Providers:
- **AWS** — X-Ray for traces, CloudWatch for logs. Requires: access_key, secret_key, region
- **Jaeger** — GRPC endpoint for traces and logs. Requires: host, port
- **Tencent** — Tencent Cloud APM. Requires: secret_id, secret_key, region

### Load config:
1. `GET /api/provider-config`
2. Queries `TraceProviderConfig.findOne({userEmail})` and `LogProviderConfig.findOne({userEmail})`
3. Merges trace and log configs, returns combined config

### Save config:
1. `POST /api/provider-config` with body: `{ traceProvider: "jaeger", jaegerTraceConfig: { host: "...", port: 16685 } }`
2. Upserts into `trace_provider_configs` and/or `log_provider_configs` collections

### Delete config:
1. `DELETE /api/provider-config?providerType=trace&provider=jaeger`
2. Either `$unset` the specific provider's config field, or delete the entire document

### How providers affect other flows:
The selected provider is stored in localStorage and read by `buildProviderParams()` in `utils/provider.ts`. Every trace/log API call includes `trace_provider`, `log_provider`, `trace_region`, `log_region` as query params.

---

## FLOW 9 — Integration Tokens

**Components**: `Item.tsx`, `RightPanel.tsx` (Integrate page) → `/api/get_connect`, `/api/post_connect`, `/api/delete_connect` → Direct MongoDB

**THIS FLOW SKIPS THE PYTHON BACKEND.**

### Supported integrations:
- **Rootix** — SDK token for OTel Collector. Stored in `rootix_tokens` collection
- **GitHub** — Personal access token for code fetch + PR creation. Stored in `connection_tokens` collection
- **Notion** — (placeholder)
- **Slack** — (placeholder)
- **OpenAI** — (placeholder, custom key)

### Fetch token:
1. `GET /api/get_connect?resourceType=rootix`
2. Queries by `user_email` (from Clerk) + `token_type` (from ResourceType)
3. Returns `{ success: true, token: "tr_abc123..." }`

### Save token:
1. `POST /api/post_connect` with body: `{ resourceType: "github", token: "ghp_xxx..." }`
2. Upserts into `connection_tokens` collection

### Delete token:
1. `DELETE /api/delete_connect` with body: `{ resourceType: "github" }`
2. Removes from `connection_tokens` collection

### User identification:
The `get_connect` route uses `user_email` from Clerk's `currentUser()`. For Rootix tokens, it queries the `rootix_tokens` collection. For other token types, it queries `connection_tokens` with both `user_email` and `token_type`.

---

## FLOW 10 — OTLP Ingestion

**Components**: OpenTelemetry Collector → Python `InternalRouter.receive_traces` (NO UI INVOLVED)

**This flow has no frontend component.** It's a backend-only endpoint that receives telemetry data from the user's application.

1. User's application is instrumented with OpenTelemetry SDK
2. OTel Collector sends traces to `POST /v1/traces` on the Python backend
3. `InternalRouter.receive_traces` receives the request
4. Payload can be protobuf binary or JSON (detected by Content-Type header)
5. Supports gzip compression (detected by Content-Encoding header)
6. Parses `ExportTraceServiceRequest` protobuf message
7. For each span in each resource span:
   - Extracts `hash` attribute (user identifier hash)
   - Counts span (1 trace unit)
   - Sums `num_debug_logs`, `num_info_logs`, `num_warning_logs`, `num_error_logs`, `num_critical_logs` (log units)
8. Accumulates in memory buffer: `{user_hash: {traces: count, logs: count}}`
9. Every 10 seconds (`FLUSH_INTERVAL_SECONDS`), flushes buffer:
   - Looks up `user_sub` from `user_hash` via MongoDB (`get_user_sub_by_hash`)
   - Calls Autumn billing tracker: `track_traces_and_logs(customer_id, trace_count, log_count)`

---

## 19. Authentication

**Provider**: Clerk (https://clerk.com)

### Frontend:
- `middleware.ts` protects routes matching `/explore(.*)`, `/integrate(.*)`, `/settings(.*)`
- Excludes: `/api/autumn(.*)`, `/sign-in`, `/sign-up`, `/pricing`, `/login`
- Clerk provides `useAuth()`, `useUser()` hooks in components
- `getToken()` retrieves JWT for API calls

### API Route Auth (Next.js → Python):
- `createBackendAuthHeaders()` in `lib/server-auth-headers.ts` creates headers:
  - `Authorization: Bearer <clerk_jwt>`
  - `Content-Type: application/json`
  - `X-User-Email: user@example.com` (from Clerk `currentUser()`)
  - `X-User-Sub: user_sub_id` (from Clerk)
- Python backend extracts these via `get_user_credentials(request)` in `rest/utils/auth.py`

### MongoDB Direct Routes:
- `auth()` and `currentUser()` from `@clerk/nextjs/server` are called directly in the route handler
- User is identified by `userEmail` from Clerk's `currentUser().emailAddresses[0].emailAddress`

---

## 20. Environment Variables

| Variable | Used By | Description |
|---|---|---|
| `REST_API_ENDPOINT` | Next.js API routes | Python FastAPI backend URL (e.g., `http://localhost:8000`) |
| `MONGODB_URI` | Next.js (direct MongoDB routes) | MongoDB connection string |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Frontend | Clerk public key |
| `CLERK_SECRET_KEY` | API routes | Clerk secret key |
| `NEXT_PUBLIC_CLERK_SIGN_IN_URL` | Frontend | Clerk sign-in page URL |
| `NEXT_PUBLIC_CLERK_SIGN_UP_URL` | Frontend | Clerk sign-up page URL |
| `DB_REASONING_COLLECTION` | Backend/Frontend | MongoDB collection for reasoning streams (default: "reasoning_streams") |
| `DB_BACKEND` | Python backend | "sqlite" or "mongodb" |
| `SQLITE_DB_PATH` | Python backend | Path to SQLite database file |
| `MONGODB_DB` | Python backend | MongoDB database name |

---

*End of Rootix UI Reference Document*
