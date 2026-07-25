"""Answer-letter extraction for OmniSpatial.

Kept in its own module (no imports beyond ``re``, no module-level side effects)
so offline re-scorers can import the *exact* deployed logic without triggering
``utils.py``'s dataset ``snapshot_download``.

The official OmniSpatial scorer only accepts a literal ``Answer: X`` line and
falls back to predicting ``"A"`` when it finds nothing. That fallback silently
converts a *format* mismatch into a *score*: a model that never emits the
required line is graded at exactly the base rate of gold-``A`` items, which
looks like a plausible-but-bad result rather than a parse failure. Reasoning
models fine-tuned to answer as ``<think>...</think> D`` hit this exactly.

So we extract in tiers and return ``None`` when nothing is found, letting the
caller score the sample incorrect rather than guessing a letter.

Deliberately *not* ``_task_utils.mcq_extract.extract_mcq_answer``: that one is a
best-effort extractor for tasks with no mandated answer format, so it ranks
positional formats above the phrase match and always returns some letter when
any appears in the text. OmniSpatial's protocol does mandate ``Answer: X``, and
"found no answer" has to stay distinguishable from "answered A". On the four
cached Qwen3-VL runs the two agree on 92-98% of samples and differ by <1pp in
accuracy, so this is about semantics, not about scores.
"""

import re

# Tier 1: the format the task prompt actually asks for. Tolerates markdown bold
# ("**Answer:** B"), parentheses ("Answer: (B)") and a full-width colon.
_EXPLICIT = re.compile(r"answer\s*[:：]\s*\**\s*\(?\s*([A-D])\b", re.IGNORECASE)

# Tier 2: the whole response is just the letter, e.g. "D", "(D)", "D." — what a
# reasoning model leaves behind once its <think> block has been stripped.
_BARE = re.compile(r"^\(?\s*([A-D])\s*[)\].,:;]?\s*$", re.IGNORECASE)

# Tier 3: last option letter mentioned as a standalone token. Case-sensitive on
# purpose — lowercase "a" is the English article, and matching it would fire on
# nearly every response.
_STANDALONE = re.compile(r"\b([A-D])\b")

_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def strip_reasoning(response: str) -> str:
    """Drop <think> blocks and unwrap <answer> tags.

    The evaluator's ``--reasoning_tags`` usually does this already; doing it
    again here is idempotent and keeps the parser correct for runs (and
    re-scorers) where that flag was not set.
    """
    text = _THINK.sub(" ", response)
    # A chat template may prefill "<think>", leaving only the closing tag.
    if "</think>" in text and "<think>" not in text:
        text = text.rsplit("</think>", 1)[-1]
    tagged = _ANSWER_TAG.findall(text)
    if tagged:
        return tagged[-1].strip()
    if "<answer>" in text.lower():  # opened but never closed (hit the token cap)
        return re.split(r"<answer>", text, flags=re.IGNORECASE)[-1].strip()
    return text.strip()


def extract_answer_letter(response: str):
    """Return the option letter the response commits to, or None if unparseable.

    None means "no answer found" and must be scored incorrect — never defaulted
    to a letter, which would make the metric depend on the gold-label
    distribution instead of the model.
    """
    if not response:
        return None
    text = strip_reasoning(response)

    explicit = _EXPLICIT.findall(text)
    if explicit:
        return explicit[-1].upper()

    bare = _BARE.match(text)
    if bare:
        return bare.group(1).upper()

    standalone = _STANDALONE.findall(text)
    if standalone:
        return standalone[-1].upper()
    return None
