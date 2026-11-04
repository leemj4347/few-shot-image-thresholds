# Condition-level criteria and edit names
from __future__ import annotations

DELTA = 0.05
BASELINE_METHOD = "DBpHash fixed"
HASH_METHOD = "Hash convention (10 bits)"


def variant_names() -> list[str]:
    import transforms as TR
    return [s[0] for s in TR.EDITS]
