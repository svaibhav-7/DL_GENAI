import re

WRAPPER_PATTERNS = [
    r"^(pick the best possible answer:|select the most accurate option:|"
    r"determine the correct option:|choose the correct answer:|identify the correct statement:)\s*",
    r"\s*(among the listed options\.?|from the following choices\.?|carefully\.?|based on the given context\.?)$",
]

def strip_wrapper(text: str) -> str:
    """
    Cleans up wrappers from text.
    """
    t = str(text).strip().lower()
    for pat in WRAPPER_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
    return t

def apk3(actual_idx, top3_idx):
    """
    Computes Average Precision at k=3 for a single query.
    """
    for i, p in enumerate(top3_idx[:3]):
        if p == actual_idx:
            return 1.0 / (i + 1)
    return 0.0
