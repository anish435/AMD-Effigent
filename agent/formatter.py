"""
Output formatting and post-processing cleanups.

Provides tools to clean conversational filler prefixes and ensure completions
are concise and to-the-point for token efficiency.
"""

from __future__ import annotations

import re

# Curated regex patterns for conversational filler openers (case-insensitive)
_FILLER_PATTERNS = [
    r"^(?:sure|certainly|of course|absolutely)[!,\s]*",
    r"^(?:here is|here's)(?:\s+the)?(?:\s+answer)?(?:\s+to\s+your\s+question)?[!:\s]*",
    r"^the\s+answer\s+is[:\s]*",
    r"^i\s+think\s+that\s*",
    r"^(?:based\s+on\s+the\s+provided\s+text|based\s+on\s+the\s+text|according\s+to\s+the\s+text)[,:\s]*",
    r"^(?:here\s+are|here's\s+a\s+list\s+of)[,:\s]*",
]

def strip_filler(text: str) -> str:
    """
    Strips common conversational filler openers from the start of the output string.
    
    Runs recursively/repeatedly to handle chained fillers like "Sure, here is the answer: 42".
    """
    cleaned = text.strip()
    
    changed = True
    while changed:
        changed = False
        for pattern in _FILLER_PATTERNS:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                # Ensure we don't completely empty out the output
                candidate = cleaned[match.end():].strip()
                if candidate:
                    cleaned = candidate
                    changed = True
                    break
                    
    # Capitalize the first letter if it got stripped down to lowercase and starts a sentence
    if cleaned and cleaned != text.strip():
        if cleaned[0].islower() and (len(cleaned) == 1 or cleaned[1].isalpha() or cleaned[1].isspace()):
            cleaned = cleaned[0].upper() + cleaned[1:]
            
    return cleaned
