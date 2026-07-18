# adapted from ai2d/utils.py
import re

from lmms_eval.filters.extraction import ExtendedRegexFilter


def _extract_choice_letter(resp):
    """Extract the MCQ option letter from a (possibly <think>-reasoned) response.

    IllusionVQA asks for a bare letter, but reasoning models emit
    '<think>...</think> B.', so the original ^\\s*([A-Z])\\. anchor on the full
    response fails. Parse the answer after </think>. Returns None to fall back to
    the raw response (non-reasoning outputs are handled by the same patterns).
    """
    seg = resp.rsplit("</think>", 1)[1].strip() if "</think>" in resp else resp.strip()
    m = re.search(r"\\boxed\{\s*([A-Z])\b", seg)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:answer|option)(?:\s+(?:is|should be))?\s*[:\-]?\s*\(?([A-Z])\b", seg, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"\(?([A-Za-z])[).:\s]", seg)
    if m:
        return m.group(1).upper()
    stripped = seg.strip("()*.\n ")
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()
    return None


def illusionvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]

    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


def illusionvqa_doc_to_target(doc):
    len_choices = len(doc["options"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["options"].index(doc["answer"])]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Process each response. Parse the option letter from the answer
            # (after </think> for reasoning models); fall back to the raw response.
            filtered = []
            for resp in r:
                letter = _extract_choice_letter(resp)
                filtered.append(letter if letter is not None else resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps
