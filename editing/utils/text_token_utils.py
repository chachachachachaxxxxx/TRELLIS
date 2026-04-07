"""Text token alignment utilities for prompt-to-prompt editing."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List


def build_text_token_metadata(
    source_cond: "torch.Tensor",
    edit_cond: "torch.Tensor",
    source_prompt: str,
    edit_prompt: str,
    tokenizer,
) -> Dict:
    """Build token metadata for text prompt-to-prompt editing.

    Args:
        source_cond: Source condition tensor
        edit_cond: Edit condition tensor
        source_prompt: Source text prompt
        edit_prompt: Edit text prompt
        tokenizer: Text tokenizer

    Returns:
        Dictionary with token alignment metadata
    """
    source_meta = _tokenize_with_metadata(source_prompt, tokenizer)
    edit_meta = _tokenize_with_metadata(edit_prompt, tokenizer)

    # Align tokens using sequence matcher
    src_content_indices = [
        idx
        for idx, token in enumerate(source_meta["display_tokens"])
        if token not in {"<|startoftext|>", "<|endoftext|>"}
    ]
    edit_content_indices = [
        idx
        for idx, token in enumerate(edit_meta["display_tokens"])
        if token not in {"<|startoftext|>", "<|endoftext|>"}
    ]

    src_content_tokens = [source_meta["display_tokens"][idx] for idx in src_content_indices]
    edit_content_tokens = [edit_meta["display_tokens"][idx] for idx in edit_content_indices]

    matcher = SequenceMatcher(None, src_content_tokens, edit_content_tokens)

    src_keep_indices: List[int] = []
    edit_keep_indices: List[int] = []
    aligned_pairs: List[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            src_full_idx = src_content_indices[i1 + offset]
            edit_full_idx = edit_content_indices[j1 + offset]
            src_keep_indices.append(src_full_idx)
            edit_keep_indices.append(edit_full_idx)
            aligned_pairs.append(
                {
                    "source_token_index": src_full_idx,
                    "edit_token_index": edit_full_idx,
                    "source_token": source_meta["display_tokens"][src_full_idx],
                    "edit_token": edit_meta["display_tokens"][edit_full_idx],
                }
            )

    src_keep_set = set(src_keep_indices)
    edit_keep_set = set(edit_keep_indices)
    edited_source_indices = [idx for idx in src_content_indices if idx not in src_keep_set]
    edited_edit_indices = [idx for idx in edit_content_indices if idx not in edit_keep_set]

    return {
        "source_prompt": source_meta["prompt"],
        "edit_prompt": edit_meta["prompt"],
        "source_tokens": source_meta["display_tokens"],
        "edit_tokens": edit_meta["display_tokens"],
        "source_keep_indices": src_keep_indices,
        "edit_keep_indices": edit_keep_indices,
        "source_keep_tokens": [source_meta["display_tokens"][idx] for idx in src_keep_indices],
        "edit_keep_tokens": [edit_meta["display_tokens"][idx] for idx in edit_keep_indices],
        "source_edited_indices": edited_source_indices,
        "edit_edited_indices": edited_edit_indices,
        "source_edited_tokens": [source_meta["display_tokens"][idx] for idx in edited_source_indices],
        "edit_edited_tokens": [edit_meta["display_tokens"][idx] for idx in edited_edit_indices],
        "aligned_pairs": aligned_pairs,
    }


def _tokenize_with_metadata(prompt: str, tokenizer) -> Dict:
    """Tokenize prompt and return metadata.

    Args:
        prompt: Text prompt
        tokenizer: Text tokenizer

    Returns:
        Dictionary with tokenization metadata
    """
    tokens = tokenizer.encode(prompt)
    display_tokens = [tokenizer.decode([token_id]) for token_id in tokens]

    return {
        "prompt": prompt,
        "tokens": tokens,
        "display_tokens": display_tokens,
    }
