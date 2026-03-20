"""
Sequential Context Chunking - Splits large trace context into LLM-sized pieces.

WHY THIS EXISTS:
LLMs have token limits (e.g., Llama 3.3 on Groq = 128k tokens).
A large trace can easily exceed this. This module splits the context
into overlapping chunks so no information is lost at chunk boundaries.

HOW IT WORKS:
1. Convert the span tree to a JSON string (the "context")
2. If context < CHUNK_SIZE characters → single chunk (most common)
3. If context > CHUNK_SIZE → split into overlapping pieces
4. Each chunk gets sent to the LLM separately
5. Results are merged by the summarizer

EXECUTION FLOW:
    context = json.dumps(tree.to_dict(), indent=2)
    chunks = get_trace_context_messages(context)
    # chunks = ["chunk1...", "chunk2..."]  (usually just 1 chunk)
"""

from typing import Iterator


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Each character ≈ 0.25 tokens, so 200k chars ≈ 50k tokens
# Groq's Llama 3.3 supports 128k tokens, so this is safe
CHUNK_SIZE = 200_000       # characters per chunk
OVERLAP_SIZE = 5_000       # character overlap between chunks


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CHUNKING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def sequential_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_size: int = OVERLAP_SIZE,
) -> Iterator[str]:
    """Split text into sequential chunks with overlap.
    
    Args:
        text: The full context string to chunk
        chunk_size: Max characters per chunk (default 200k)
        overlap_size: Characters to duplicate at chunk boundaries (default 5k)
    
    Returns:
        Iterator yielding string chunks
    
    Example:
        text = "A" * 500_000  (500k characters)
        chunk_size = 200_000
        overlap_size = 5_000
        
        Chunk 1: chars[0 : 200_000]           (200k chars)
        Chunk 2: chars[195_000 : 395_000]     (200k chars, 5k overlap)
        Chunk 3: chars[390_000 : 500_000]     (110k chars, 5k overlap)
    """
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size.")
    
    step_size = chunk_size - overlap_size
    for i in range(0, len(text), step_size):
        yield text[i:i + chunk_size]


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API (used by SingleRCAAgent)
# ═══════════════════════════════════════════════════════════════════════════════

def get_trace_context_messages(
    context: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_size: int = OVERLAP_SIZE,
) -> list[str]:
    """Convert trace context into a list of message chunks.
    
    This is the function imported by SingleRCAAgent:
        from src.agents.chunk.sequential import get_trace_context_messages
    
    EXECUTION FLOW:
    1. Takes the full JSON context string
    2. Wraps each chunk with a header indicating its position
    3. Returns list of formatted chunks
    
    Example:
        context = json.dumps(tree_dict, indent=2)  # 150k chars
        messages = get_trace_context_messages(context)
        # len(messages) == 1  (fits in single chunk)
        # messages[0] == "=== Trace Context (Part 1/1) ===\n{...}"
    """
    chunks = list(sequential_chunk(context, chunk_size, overlap_size))
    total = len(chunks)
    
    messages = []
    for i, chunk in enumerate(chunks):
        header = f"=== Trace Context (Part {i + 1}/{total}) ==="
        if total > 1:
            header += (
                f"\n[Note: This is chunk {i + 1} of {total}. "
                f"There may be overlap with adjacent chunks.]"
            )
        messages.append(f"{header}\n\n{chunk}")
    
    return messages
