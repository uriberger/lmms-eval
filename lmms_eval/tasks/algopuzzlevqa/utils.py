"""Utilities for the AlgoPuzzleVQA task.

Dataset: declare-lab/AlgoPuzzleVQA (https://huggingface.co/datasets/declare-lab/AlgoPuzzleVQA)
Paper:  "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil
         Serious Challenges in Multimodal Reasoning" (ACL 2024, declare-lab).

Each instance is a multiple-choice question over an image.  The dataset stores
``options`` as a *pre-shuffled* list of answer strings (variable length, 2..N)
and ``answer`` as the option *text* (not a letter).  We render the options as
lettered choices (A, B, C, ...) at prompt time and map the gold answer text back
to its letter at scoring time.
"""

import string

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer


def _letters(n):
    """Return the first ``n`` uppercase choice letters (A, B, C, ...)."""
    return list(string.ascii_uppercase[:n])


def algopuzzlevqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def algopuzzlevqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"].strip()
    options = doc["options"]
    letters = _letters(len(options))
    options_block = "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options))

    return f"{pre_prompt}{question}\n{options_block}{post_prompt}"


def _gold_letter(doc):
    """Map the gold answer *text* to its choice letter (case-insensitive fallback)."""
    options = [str(opt).strip() for opt in doc["options"]]
    answer = str(doc["answer"]).strip()
    letters = _letters(len(options))

    for letter, opt in zip(letters, options):
        if opt == answer:
            return letter
    for letter, opt in zip(letters, options):
        if opt.lower() == answer.lower():
            return letter
    eval_logger.warning(f"AlgoPuzzleVQA: gold answer {answer!r} not found in options {options!r}")
    return ""


def algopuzzlevqa_process_results(doc, results):
    """Score a single prediction.

    Primary path: extract the choice letter from the model output and compare to
    the gold letter.  Fallback: some models answer with the option text directly
    (e.g. "No" for a Yes/No puzzle), so we also accept a normalized exact match
    against the gold answer text.
    """
    pred = results[0]
    options = [str(opt).strip() for opt in doc["options"]]
    letters = _letters(len(options))
    gold_letter = _gold_letter(doc)

    pred_letter = extract_mcq_answer(pred, choices=letters)

    score = 0.0
    if pred_letter and gold_letter and pred_letter == gold_letter:
        score = 1.0
    else:
        # Fallback: model echoed the option text instead of a letter.
        def _norm(s):
            return str(s).strip().strip(".,!?;:'\"").lower()

        if _norm(pred) == _norm(doc["answer"]):
            score = 1.0

    return {"exact_match": score}
