"""Remove a benchmark's "answer directly" instruction from a rendered prompt.

Off unless LMMS_STRIP_ANSWER_FORMAT_INSTRUCTION is set to 1/true/yes.

Why this exists
---------------
Most multiple-choice tasks end their prompt with a sentence telling the model to
reply with nothing but the option letter -- "Answer with the option's letter from
the given choices directly.", "Note: You only need to respond with A, B, C, D
without providing any additional information.". That instruction sits in the USER
turn, so it beats a system prompt asking for a reasoning chain: it is closer and
more specific. The measured effect on a reasoning-trained Qwen3-VL-8B is total --
median response length 1 character, not one response in 2,638 containing a
<think> block. Whatever the RL training bought, the benchmark cannot see it.

Papers that report reasoning-mode numbers (e.g. EASE, arXiv 2605.30912) append
the opposite instruction, so their figures are not comparable with a run that
leaves these sentences in place. This makes that difference switchable.

Why it edits the rendered string rather than lmms_eval_specific_kwargs
---------------------------------------------------------------------
Blanking pre_prompt/post_prompt in the task config would be tidier, and it is
also not sufficient. hrbench_doc_to_text hardcodes "Answer the option letter
directly." in an f-string, and vstar-bench ships the sentence inside the
dataset's own `text` column -- which is why it appears TWICE in a V* prompt, once
from the data and once from post_prompt. Only the final string has them all.

The patterns are an explicit list, not a general "sentence mentioning 'directly'"
rule, because a false positive here silently rewrites the question. Add to the
list when a benchmark needs it; the stripper reports what it removed, once per
task, so a prompt that still carries one is visible rather than assumed gone.
"""

import os
import re

from loguru import logger as eval_logger

_ENV = "LMMS_STRIP_ANSWER_FORMAT_INSTRUCTION"

# Whole sentences only, each anchored so it cannot eat part of a question.
_PATTERNS = [
    # vstar_bench (post_prompt, and again inside the dataset's `text` column)
    r"[ \t]*Answer with the option'?s letter from the given choices directly\.?",
    # hrbench4k / hrbench8k (hardcoded in hrbench_doc_to_text)
    r"[ \t]*Answer the option letter directly\.?",
    # cv_bench (pre_prompt; the {} is already formatted to "A, B, C, D" by here)
    r"[ \t]*Note:\s*You only need to respond with[^\n.]*?without providing any additional information\.?",
    # common across mmbench / realworldqa / scienceqa / mmmu_pro variants
    r"[ \t]*Answer the question using a single word or phrase\.?",
    r"[ \t]*Answer with the option'?s letter from the given choices\.?",
    r"[ \t]*Please answer the question (?:directly )?with (?:only )?the option'?s? letter[^\n.]*\.?",
    r"[ \t]*Please (?:select|respond with|answer with) the correct (?:option|answer|letter)[^\n.]*directly\.?",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

_reported: set[str] = set()


def strip_answer_format_enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def strip_answer_format_instructions(text: str, task: str = "") -> str:
    """Drop terse-answer instructions from `text`. No-op unless the env var is set.

    `task` only labels the one-time log line, so a run makes it obvious which
    benchmarks were actually rewritten and which were left untouched.
    """
    if not text or not strip_answer_format_enabled():
        return text

    removed = []
    out = text
    for pattern in _COMPILED:
        out, n = pattern.subn("", out)
        if n:
            removed.append(f"{pattern.pattern[:40]}...x{n}")

    if not removed:
        return text

    # Blank lines and trailing space left where the sentence used to be.
    out = re.sub(r"[ \t]+(\n)", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()

    key = task or "<unknown task>"
    if key not in _reported:
        _reported.add(key)
        eval_logger.info(f"[{_ENV}] {key}: removed {len(removed)} answer-format instruction(s) per prompt")
    return out
