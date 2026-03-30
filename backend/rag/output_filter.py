# =====================================================
# rag/output_filter.py
# Filters LLM output before sending to user.
#
# WHY filter the OUTPUT too?
# Even with a strong system prompt, LLMs can sometimes:
# 1. Leak system prompt fragments
# 2. Generate content not in the provided context
# 3. Reveal role/access information
#
# This is the final security gate before the response
# reaches the user.
# =====================================================

import re
from loguru import logger


# Patterns that suggest the LLM leaked system internals
LEAKAGE_PATTERNS = [
    r"system\s+prompt",
    r"STRICT\s+RULES",
    r"CONTEXT:",
    r"\[Source\s+\d+:",          # our context formatting leaked
    r"role_access",
    r"document_chunks",
    r"rbac",
    r"supabase",
    r"access\s+restricted\s+or\s+not\s+available.*access\s+restricted",  # repeated
]

COMPILED_LEAKAGE = [
    re.compile(p, re.IGNORECASE)
    for p in LEAKAGE_PATTERNS
]


def filter_output(response: str) -> tuple[str, bool]:
    """
    Check LLM response for potential data leakage.

    Args:
        response: Raw LLM response text

    Returns:
        (filtered_response, was_filtered)
        was_filtered=True means we modified/blocked the response
    """
    if not response:
        return "Access Restricted or Not Available", True

    # Check for leakage patterns
    for pattern in COMPILED_LEAKAGE:
        if pattern.search(response):
            logger.warning(
                f"Output filtered: leakage pattern detected "
                f"pattern='{pattern.pattern[:50]}'"
            )
            return (
                "I cannot provide that information. "
                "Please contact your administrator.",
                True
            )

    # Check response isn't suspiciously long
    # (might indicate context dump)
    if len(response) > 4000:
        logger.warning(f"Response truncated: too long ({len(response)} chars)")
        response = response[:4000] + "\n\n[Response truncated]"
        return response, True

    return response, False