"""MMK12 (arXiv 2503.07365, MM-EUREKA) -- the benchmark's own 2,000-item test split.

Scoring is identical to the neighbouring `k12` task, so it is imported rather
than copied: a binary LLM judge over (question, gold answer, prediction). That
judge is what makes the task safe for reasoning models -- it reads the whole
response, so a `<think>...</think>` preamble costs nothing, unlike the regex
extractors other tasks use.

The two tasks are NOT the same data. `k12` points at lmms-lab-encoder/k12: 500
rows, a single `train` split, no `subject` column, math only, and neither of two
sampled ids appears in either MMK12 split. This one is FanqingM/MMK12 `test`:
2,000 multiple-choice questions, 500 each of math, physics, chemistry and
biology, with the options written inline in `question`.
"""

from lmms_eval.tasks.k12.utils import (
    k12_doc_to_text,
    k12_doc_to_visual,
    k12_process_results,
)

mmk12_doc_to_visual = k12_doc_to_visual
mmk12_doc_to_text = k12_doc_to_text
mmk12_process_results = k12_process_results
