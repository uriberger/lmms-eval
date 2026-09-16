"""Answer extraction for the shared `*_reasoning` scorer.

Two of these regressions were found by scoring, not by reading: LogicVista read
15.8% for Qwen3-VL-8B-Instruct where it should read 52.5%, because the parser
mis-handles exactly the shape a model produces when it is NOT using the
<think>/<answer> template. A model that replies with a bare "C" was never
affected, which is why the bug survived a whole benchmark suite.
"""

import pytest

from lmms_eval.tasks._task_utils.reasoning_utils import (
    compute_score,
    extract_anwser_tag,
    parse_mcq,
)


@pytest.mark.parametrize(
    "response,expected",
    [
        # The cue word is not the answer. Scanning from the cue's own start
        # returned the "A" of "Answer", the "F" of "Final", the "C" of
        # "Correct".
        ("**Answer: E**", "E"),
        ("Answer: E", "E"),
        ("Final Answer: **D**", "D"),
        ("### Final Answer: **B**", "B"),
        ("✅ **Correct Answer: D. All dogs can swim.**", "D"),
        ("The correct answer is: **(C) Neither set A nor set B**", "C"),
        ("**Answer: (B) False**", "B"),
        ("I choose (B)", "B"),
        ("my answer is A", "A"),
        # A leading word must not outrank the cue that follows it. "start" has
        # the highest priority of any pattern, so unrestricted it made this "B".
        ("Based on the pattern, the answer is C", "C"),
        # The shapes that already worked, kept working.
        ("E", "E"),
        ("(E)", "E"),
        ("E.", "E"),
        ("B. 8", "B"),
        ("D) Some inches are yards", "D"),
    ],
)
def test_parse_mcq_reads_the_letter_not_the_cue(response, expected):
    assert parse_mcq(response) == expected


def test_extract_prefers_the_answer_line_over_a_stray_number():
    # The numeric fallback walks backwards over EVERY line, so a chain whose
    # final line states its letter was extracted as the last number above it.
    chain = "Step 1: there are 9 shapes.\nStep 2: the count is 12.\n**Answer: A**"
    assert parse_mcq(extract_anwser_tag(chain)) == "A"


def test_extract_spans_a_cue_and_letter_on_separate_lines():
    chain = "reasoning\n### Final Answer:\n**(C) 500 increase** ✅"
    assert parse_mcq(extract_anwser_tag(chain)) == "C"


def test_extract_leaves_a_numeric_answer_alone():
    assert extract_anwser_tag("some prose\nThe total is 204").strip() == "204"


@pytest.mark.parametrize(
    "tag,body",
    [("answer", "<answer>C</answer>"), ("boxed", "the value is \\boxed{C}")],
)
def test_tagged_answers_still_win(tag, body):
    assert parse_mcq(extract_anwser_tag(f"lots of reasoning\n{body}")) == "C"


def test_free_form_and_tagged_chains_agree_on_the_same_choice():
    """The point of the fix: the format must not decide the score."""
    ground_truth = "D"
    tagged = "<think>work</think>\nD"
    free_form = (
        "Let me check each option.\n"
        "(A) is ruled out, (B) contradicts the figure, (C) is close but wrong.\n"
        "✅ **Final Answer: D**"
    )
    for solution in (tagged, free_form):
        scored = compute_score("logicvista", solution, ground_truth, {"question": "q"})
        assert scored["acc_score"] == 1.0, solution
