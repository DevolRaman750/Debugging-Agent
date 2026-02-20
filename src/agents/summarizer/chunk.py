"""
Chunk Summarizer - Merges multi-chunk LLM responses into a single answer.

WHY THIS EXISTS:
When a trace is too large for one LLM call, we split it into chunks (via
sequential.py) and send each chunk separately. Each chunk produces its own
answer. This module combines them into one coherent final response.

WHEN IS THIS USED:
- MOST of the time: trace fits in 1 chunk → this is SKIPPED (no merging needed)
- RARELY: very large traces with 100k+ logs → multiple chunks → needs merging

EXECUTION FLOW (when multiple chunks):
    1. Chunk 1 → LLM → answer_1 + refs_1
    2. Chunk 2 → LLM → answer_2 + refs_2
    3. chunk_summarize([answer_1, answer_2], [refs_1, refs_2])
       → Sends both answers to LLM asking it to merge them
       → Returns single ChatOutput with unified answer + references

ADAPTED FOR GROQ: Uses tool-calling instead of OpenAI's responses.parse()
"""

import json
from openai import AsyncOpenAI
from src.agents.types import ChatOutput, Reference
from src.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARIZER PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SUMMARIZER_SYSTEM_PROMPT = """You are a helpful TraceRoot.AI assistant that summarizes multiple response chunks into a single coherent answer.

Rules:
1. Combine all answers into one concise, clear response.
2. Include ALL information from all chunks — don't drop anything.
3. Re-number references sequentially: if chunk 1 has [1],[2],[3] and chunk 2 has [1],[2], the merged result should have [1],[2],[3],[4],[5].
4. Do NOT mention "chunks" in the final answer.
5. Be confident — don't say "data is insufficient".
6. Match references to the correct parts of the answer."""


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMA (for Groq structured output)
# ═══════════════════════════════════════════════════════════════════════════════
SUMMARIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "provide_summarized_response",
        "description": "Provide the merged/summarized response from multiple chunks",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "The merged answer combining all chunks. "
                        "Use [1], [2], etc. for references."
                    )
                },
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["span", "log", "code"],
                                "description": "Type of reference"
                            },
                            "span_id": {
                                "type": "string",
                                "description": "Span ID if applicable"
                            },
                            "log_message": {
                                "type": "string",
                                "description": "Log message if applicable"
                            },
                            "line_number": {
                                "type": "integer",
                                "description": "Line number if applicable"
                            }
                        },
                        "required": ["type"]
                    },
                    "description": "List of all references, re-numbered sequentially"
                }
            },
            "required": ["answer", "references"]
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK SUMMARIZER
# ═══════════════════════════════════════════════════════════════════════════════

async def chunk_summarize(
    response_answers: list[str],
    response_references: list[list[Reference]] | None = None,
    client: AsyncOpenAI = None,
    model: str = GROQ_MODEL,
    groq_api_key: str | None = None,
) -> ChatOutput:
    """Merge multiple chunk responses into a single coherent answer.
    
    Args:
        response_answers: List of answer strings, one per chunk
        response_references: List of reference lists, one per chunk
        client: Pre-configured AsyncOpenAI client (pointed at Groq)
        model: Model to use for summarization
        groq_api_key: Groq API key (if client not provided)
    
    Returns:
        ChatOutput with merged answer and re-numbered references
    
    EXECUTION FLOW:
    1. Format all chunk answers and references into a user message
    2. Send to Groq LLM with summarizer prompt
    3. Parse tool call response into ChatOutput
    4. Return unified answer
    """
    # Create client if not provided
    if client is None:
        api_key = groq_api_key or GROQ_API_KEY
        if not api_key:
            raise ValueError("Either client or groq_api_key must be provided")
        client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Format the chunk answers and references
    # ═══════════════════════════════════════════════════════════════════════
    formatted_chunks = []
    for i, answer in enumerate(response_answers):
        chunk_text = f"--- CHUNK {i + 1} ---\n"
        chunk_text += f"Answer:\n{answer}\n"
        
        if response_references and i < len(response_references):
            refs = response_references[i]
            if refs:
                ref_strs = [json.dumps(r.model_dump(), indent=2) for r in refs]
                chunk_text += f"References:\n{chr(10).join(ref_strs)}\n"
            else:
                chunk_text += "References: []\n"
        
        formatted_chunks.append(chunk_text)
    
    user_content = (
        f"Here are {len(response_answers)} chunk responses to merge:\n\n"
        + "\n".join(formatted_chunks)
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Call Groq LLM to merge
    # ═══════════════════════════════════════════════════════════════════════
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            tools=[SUMMARIZE_TOOL],
            tool_choice={"type": "function", "function": {"name": "provide_summarized_response"}},
            temperature=0.5,
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Parse the tool call response
        # ═══════════════════════════════════════════════════════════════════
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            args = json.loads(tool_calls[0].function.arguments)
            
            # Parse references
            refs = []
            for ref_data in args.get("references", []):
                refs.append(Reference(
                    type=ref_data.get("type", "span"),
                    span_id=ref_data.get("span_id"),
                    log_message=ref_data.get("log_message"),
                    line_number=ref_data.get("line_number"),
                ))
            
            return ChatOutput(
                answer=args["answer"],
                reference=refs,
            )
    
    except Exception as e:
        print(f"Chunk summarization error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FALLBACK: Simple concatenation if LLM fails
    # ═══════════════════════════════════════════════════════════════════════
    combined_answer = "\n\n---\n\n".join(response_answers)
    combined_refs = []
    if response_references:
        for refs in response_references:
            if refs:
                combined_refs.extend(refs)
    
    return ChatOutput(
        answer=combined_answer,
        reference=combined_refs,
    )
