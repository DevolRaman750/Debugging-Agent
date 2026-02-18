"""
Feature Selection - Decides which log/span fields are relevant to the user's question.

WHAT THIS DOES:
When user asks "why is my API slow?", the LLM doesn't need to see log file names
or line numbers — it just needs latencies and error messages. This module asks the
LLM to pick only the relevant fields, which reduces the context size significantly.

ADAPTED FOR GROQ: Uses tool-calling instead of OpenAI's responses.parse() API.
"""

import json
from openai import AsyncOpenAI
from src.agents.types import LogFeature, SpanFeature

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════
LOG_FEATURE_SELECTOR_PROMPT = (
    "You are a helpful assistant that selects relevant log features "
    "based on the user's message.\n"
    "Available log features:\n"
    "- log utc timestamp\n"
    "- log level\n"
    "- file name\n"
    "- function name\n"
    "- log message value\n"
    "- line number\n"
    "- log line source code\n\n"
    "Select ONLY the features needed to answer the user's question. "
    "Always include 'log message value'. "
    "Include 'log level' if user asks about errors/warnings."
)

SPAN_FEATURE_SELECTOR_PROMPT = (
    "You are a helpful assistant that selects relevant span features "
    "based on the user's message.\n"
    "Available span features:\n"
    "- span latency\n"
    "- span utc start time\n"
    "- span utc end time\n\n"
    "Select ONLY the features needed to answer the user's question. "
    "Include 'span latency' if user asks about performance/slowness. "
    "Include timestamps if user asks about timing/ordering."
)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS (for Groq tool-calling)
# ═══════════════════════════════════════════════════════════════════════════════
LOG_FEATURE_TOOL = {
    "type": "function",
    "function": {
        "name": "select_log_features",
        "description": "Select the relevant log features for the user's query",
        "parameters": {
            "type": "object",
            "properties": {
                "log_features": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [f.value for f in LogFeature]
                    },
                    "description": "List of relevant log features"
                }
            },
            "required": ["log_features"]
        }
    }
}

SPAN_FEATURE_TOOL = {
    "type": "function",
    "function": {
        "name": "select_span_features",
        "description": "Select the relevant span features for the user's query",
        "parameters": {
            "type": "object",
            "properties": {
                "span_features": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [f.value for f in SpanFeature]
                    },
                    "description": "List of relevant span features"
                }
            },
            "required": ["span_features"]
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTORS
# ═══════════════════════════════════════════════════════════════════════════════

async def log_feature_selector(
    user_message: str,
    client: AsyncOpenAI,
    model: str = GROQ_MODEL,
) -> list[LogFeature]:
    """Ask the LLM which log features are relevant to the user's question.
    
    EXECUTION FLOW:
    1. Send user message + available features to Groq LLM
    2. LLM picks relevant features via tool calling
    3. Parse the tool call response into LogFeature enums
    4. Return the list of features
    
    EXAMPLE:
        User: "Show me all error logs"
        LLM picks: [LOG_LEVEL, LOG_MESSAGE_VALUE]
        
        User: "What function threw the exception?"
        LLM picks: [LOG_LEVEL, LOG_MESSAGE_VALUE, LOG_FUNC_NAME, LOG_FILE_NAME]
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LOG_FEATURE_SELECTOR_PROMPT},
                {"role": "user", "content": user_message}
            ],
            tools=[LOG_FEATURE_TOOL],
            tool_choice={"type": "function", "function": {"name": "select_log_features"}},
            temperature=0.3,
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            args = json.loads(tool_calls[0].function.arguments)
            features = []
            for feature_str in args.get("log_features", []):
                try:
                    features.append(LogFeature(feature_str))
                except ValueError:
                    continue  # Skip invalid feature names
            
            # Always include log message value as minimum
            if LogFeature.LOG_MESSAGE_VALUE not in features:
                features.append(LogFeature.LOG_MESSAGE_VALUE)
            
            return features

    except Exception as e:
        print(f"Log feature selection error: {e}")

    # Fallback: return essential features
    return [
        LogFeature.LOG_LEVEL,
        LogFeature.LOG_MESSAGE_VALUE,
        LogFeature.LOG_FUNC_NAME,
    ]


async def span_feature_selector(
    user_message: str,
    client: AsyncOpenAI,
    model: str = GROQ_MODEL,
) -> list[SpanFeature]:
    """Ask the LLM which span features are relevant to the user's question.
    
    EXECUTION FLOW:
    1. Send user message + available features to Groq LLM
    2. LLM picks relevant features via tool calling
    3. Parse the tool call response into SpanFeature enums
    4. Return the list of features
    
    EXAMPLE:
        User: "Why is this trace slow?"
        LLM picks: [SPAN_LATENCY]
        
        User: "What happened between 3pm and 4pm?"
        LLM picks: [SPAN_UTC_START_TIME, SPAN_UTC_END_TIME]
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SPAN_FEATURE_SELECTOR_PROMPT},
                {"role": "user", "content": user_message}
            ],
            tools=[SPAN_FEATURE_TOOL],
            tool_choice={"type": "function", "function": {"name": "select_span_features"}},
            temperature=0.3,
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            args = json.loads(tool_calls[0].function.arguments)
            features = []
            for feature_str in args.get("span_features", []):
                try:
                    features.append(SpanFeature(feature_str))
                except ValueError:
                    continue

            # Always include latency as minimum
            if SpanFeature.SPAN_LATENCY not in features:
                features.append(SpanFeature.SPAN_LATENCY)

            return features

    except Exception as e:
        print(f"Span feature selection error: {e}")

    # Fallback: return all span features
    return [
        SpanFeature.SPAN_LATENCY,
        SpanFeature.SPAN_UTC_START_TIME,
        SpanFeature.SPAN_UTC_END_TIME,
    ]
