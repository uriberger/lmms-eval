"""DailyClue (Crysun/DailyClue) task for lmms-eval.

Faithful port of the official evaluation protocol from
https://github.com/xiaominli1020/DailyClue (infer/inference.py + eval/eval.py).

Prompting (get_system_prompt / build_user_prompt) supports the three official
prompt modes; the model's system+user prompt are folded into a single
doc_to_text string because lmms-eval sets the system prompt globally, not
per-sample. The final answer is extracted from the <answer>...</answer> tag,
then scored per-category/per-format exactly as eval.py:

  - Location Identification (open-ended): verify_location -- country exact +
    region fuzzy match (Levenshtein < 3 per word, '/'-separated GT aliases).
  - Multiple choice: judge_answer_choice -- exact or leading A-D letter match.
  - Yes or No: judge_answer_yes_no -- exact or contains match.
  - Science / Daily Commonsense open-ended: judge_answer_open_ended -- exact /
    numeric / date shortcuts, else LLM-as-judge (default gpt-4o-mini).

Judge runs through the NVIDIA inference gateway (same setup as ../saliency_r1):
env OPENAI_API_KEY|NVIDIA_API_KEY, OPENAI_BASE_URL (default the NVIDIA gateway),
JUDGE_MODEL (default "azure/openai/gpt-4o-mini"). Prompt mode via
lmms_eval_specific_kwargs.mode (default "c").
"""
import os
import random
import re
import time
from collections import defaultdict

import Levenshtein
from loguru import logger as eval_logger
from openai import OpenAI

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
# LLM judge via the NVIDIA inference gateway (same convention as ../saliency_r1
# trl/rewards/openai_rewards.py): key from OPENAI_API_KEY|NVIDIA_API_KEY, base
# url from OPENAI_BASE_URL, and provider-prefixed model names (the bare
# "gpt-4o-mini" alias returns 403 key_model_access_denied). Override JUDGE_MODEL.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "azure/openai/gpt-4o-mini")
_judge_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY") or ""
_judge_base_url = os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com")
_judge_client = OpenAI(api_key=_judge_api_key or "EMPTY", base_url=_judge_base_url)

NUMERIC_ONLY_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
MONTH_PATTERN = re.compile(
    r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
    r"august|aug|september|sept|sep|october|oct|november|nov|december|dec)\b",
    re.IGNORECASE,
)
DAY_PATTERN = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?")

# canonical category_1 label -> stable metric name used in the YAML metric_list.
CATEGORY_TO_METRIC = {
    "Daily Commonsense Reasoning": "daily_commonsense",
    "Location Identification": "location",
    "Spatial Reasoning": "spatial",
    "Science": "science",
}

# ----------------------------------------------------------------------------
# prompt construction (ported from infer/inference.py)
# ----------------------------------------------------------------------------
_LOCATION_FEWSHOT = [
    """<think><image clue>1: The text on the sign "Etn..." or "Edo". <image clue>2: The pedestrian traffic sign. <image clue>3: The blue glass corner-cut design of the Titanium La Portada building. <image clue>4: Wide streets and bus lanes.
Based on the visual clues, Clue 3 identifies the Titanium La Portada building, which is a landmark in the 'El Golf'/'Sanhattan' area of the Santiago financial district. Clue 4 suggests this is likely Avenida Apoquindo. Although the text in Clue 1 is not perfectly clear, the combination of the unique building architecture and the street layout strongly points to Santiago. Previously, the user predicted locations in the Santiago Metropolitan Region. Therefore, this picture shows an area in this region.</think> <answer>$Chile,Santiago Metropolitan Region$</answer>""",
    """<think><image clue>1: The sandstone temple complex. <image clue>2: The rock-cut stela. <image clue>3: The desert sand environment. <image clue>4: Hieroglyphic carvings.
The visual clues show a sandstone temple with hieroglyphics and a rock-cut stela (Clues 1, 2, 4) located in a desert environment (Clue 3). These features are characteristic of the Abu Simbel temples. The location is near Lake Nasser, which is in the Aswan Governorate of Egypt.</think> <answer>$Egypt,Aswan Governorate$</answer>""",
    """<think><image clue>1: The Atatürk Cultural Center (modern rectangular building). <image clue>2: The white flagpoles. <image clue>3: The tram line above the street. <image clue>4: Gray brick paved ground.
Starting with the clues, the modern building in Clue 1 resembles the Atatürk Cultural Center. The presence of the tram line (Clue 3) and the specific paving (Clue 4) matches the humanistic characteristics of Taksim Square. The white flagpoles (Clue 2) are also consistent with this location. On the right are modern apartments, left are older buildings, conforming to the square's layout.</think> <answer>$Turkey,Istanbul$</answer>""",
]


def _clean_few_shot_example(example, mode):
    if mode == "a":
        return re.sub(r"<think>.*?</think>\s*", "", example, flags=re.DOTALL).strip()
    elif mode == "b":
        match = re.search(r"(<think>)(.*?)(</think>)", example, re.DOTALL)
        if match:
            start_tag, content, end_tag = match.groups()
            content_clean = re.sub(r"<image clue>.*?(?=(\n|Based on|Starting with|The visual clues))", "", content, flags=re.DOTALL)
            content_clean = re.sub(r"<image clue>\d+:.*?(?=(\.|$))", "", content_clean, flags=re.DOTALL)
            return f"{start_tag}{content_clean.strip()}{end_tag} {example.split('</think>')[-1]}"
        return example
    return example


def _location_fewshot_prompt(mode):
    demo = "\nBelow are some examples of how to response to the user's question.\n"
    for ex in _LOCATION_FEWSHOT:
        demo += _clean_few_shot_example(ex, mode) + "\n\n"
    return demo


def get_system_prompt(category, format_type, mode):
    category = (category or "").lower()
    format_type = (format_type or "").lower()
    base_role = "You are an expert AI assistant with advanced visual reasoning capabilities."
    strict = "CRITICAL: You MUST MUST output the final answer wrapped in <answer>...</answer> tags at the very end. Ensure the format is exactly <answer>Your Answer</answer>."

    if mode == "a":
        instruction = "CRITICAL INSTRUCTION: Do not use <thinking> tags or think under any circumstances. Provide the response immediately as if your thinking budget is set to 0. Do not use reasoning steps."
        if category == "location identification":
            fmt = "Provide ONLY the location using the format: <answer>country,administrative_area_level_1</answer>. The administrative_area_level_1 must be the full, formal name."
        elif format_type in ["multiple choice", "yes or no"]:
            fmt = "Provide ONLY the option letter (A, B, C...) or Yes/No in the format: <answer>your answer</answer>."
        else:
            fmt = "Provide ONLY the answer using the format: <answer>your answer</answer>."
        return f"{base_role} {instruction} {fmt} {strict}"

    if mode == "b":
        instruction = "You should first reason step by step in your mind to solve the problem. Describe your reasoning naturally. Wrap all your reasoning process inside <think></think> tags."
        if category == "location identification":
            fmt = "After the </think> tag, provide ONLY the location using the format: <answer>country,administrative_area_level_1</answer>."
        elif format_type in ["multiple choice", "yes or no"]:
            fmt = "After the </think> tag, provide ONLY the option letter or Yes/No in the format: <answer>your answer</answer>."
        else:
            fmt = "After the </think> tag, provide ONLY the answer using the format: <answer>your answer</answer>."
        return f"{base_role} {instruction} {fmt} {strict}"

    # mode "c" (default, official headline clue-first reasoning)
    base_c = "You are a helpful assistant skilled at solving problems with step-by-step reasoning. First, reason step by step in your mind and use visual clues from the image as needed.Put ALL reasoning inside a single pair of <think></think> tags."
    if category == "location identification":
        fmt = "The final answer MUST be in the format: <answer>country,administrative_area_level_1</answer>. The administrative_area_level_1 must be the full, formal name of the first-level administrative region. Output nothing else (no extra words, punctuation, or whitespace outside the tags)"
    elif format_type in ["multiple choice", "yes or no"]:
        fmt = "The final answer MUST be only the option letter (e.g., A, B, C) or Yes/No, in the format: <answer>your answer</answer>."
    else:
        fmt = "The final answer MUST be in the format: <answer>your answer</answer>."
    return f"{base_c} {fmt} {strict}"


def build_user_prompt(category, question, mode):
    if (category or "").lower() == "location identification":
        location_question = "In which country and within which first-level administrative region of that country was this picture taken? Please answer in the format of <answer>country,administrative_area_level_1</answer>?"
        return _location_fewshot_prompt(mode) + location_question
    return f"Question: {question}"


def extract_answer_content(content):
    """Return the text inside the first <answer>...</answer> tag, or '' if absent."""
    try:
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        return match.group(1).strip() if match else ""
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# scoring (ported verbatim from eval/eval.py)
# ----------------------------------------------------------------------------
def _extract_month_day(text):
    if not text:
        return None
    m, d = MONTH_PATTERN.search(text), DAY_PATTERN.search(text)
    if not m or not d:
        return None
    try:
        return m.group(1).lower()[:3], int(d.group(1))
    except ValueError:
        return None


def _standardize_region_name(region):
    region = region.lower().strip()
    for suffix in [" province", " sheng", " län", " oblast", " governorate", " city", " province", " state", " territory", " region", " district"]:
        if region.endswith(suffix):
            region = region[: -len(suffix)]
    return region.strip()


def _compare_region_words(pred_words, true_words):
    if not pred_words or not true_words:
        return False
    matched = set()
    for pw in pred_words:
        found = False
        for i, tw in enumerate(true_words):
            if i in matched:
                continue
            if Levenshtein.distance(pw, tw) < 3:
                matched.add(i)
                found = True
                break
        if not found:
            return False
    return len(matched) == len(true_words)


def verify_location(answer, ground_truth):
    try:
        pred_parts = [x.strip().strip("$").lower() for x in answer.split(",")]
        if len(pred_parts) < 2:
            return 0.0, "format_error_incomplete"
        pred_country, pred_region = pred_parts[0], pred_parts[1]
        gt_parts = [x.strip().lower() for x in ground_truth.split(",")]
        true_country, true_regions = gt_parts[0], gt_parts[1]
    except (ValueError, IndexError):
        return 0.0, "mismatched_format"
    if pred_country != true_country:
        return 0.0, "mismatched_country"
    pred_region = _standardize_region_name(pred_region)
    pred_words = [x.strip() for x in pred_region.split()]
    for true_region in true_regions.split("/"):
        true_region = _standardize_region_name(true_region)
        true_words = [x.strip() for x in true_region.split()]
        if _compare_region_words(pred_words, true_words):
            return 1.0, "exact_match"
    return 0.0, "mismatched_region"


def judge_answer_choice(predicted_answer, ground_truth):
    pred_clean = predicted_answer.strip().upper()
    gt_clean = ground_truth.strip().upper()
    if pred_clean == gt_clean:
        return 1.0, "exact_match"
    gt_match = re.match(r"^\(?([A-D])\)?", gt_clean)
    pred_match = re.match(r"^\(?([A-D])\)?", pred_clean)
    if gt_match and pred_match and gt_match.group(1) == pred_match.group(1):
        return 1.0, "choice_match"
    return 0.0, "mismatched"


def judge_answer_yes_no(predicted_answer, ground_truth):
    pred_clean = predicted_answer.strip().lower()
    gt_clean = ground_truth.strip().lower()
    if pred_clean == gt_clean:
        return 1.0, "exact_match"
    if not pred_clean:
        return 0.0, "mismatched_empty_prediction"
    if gt_clean not in ["yes", "no"]:
        return 0.0, "mismatched_invalid_gt"
    if gt_clean in pred_clean:
        return 1.0, "contains_match"
    return 0.0, "mismatched"


def _judge_chat_prompt(predicted_answer, ground_truth, question):
    chat_template = """
Below are two answers to a common sense reasoning question. [Question] is the question, [Standard Answer] is the ground truth answer, and [Model Answer] is the answer predicted by a model. Determine whether these two answers are consistent.

Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same in meaning. Consider the following cases:
1. Exact match: "December 22nd" and "December 22nd"
2. Equivalent expressions: "Yes" and "Yes, it is", "No" and "No, it is not"
3. Different formats but same meaning: "14" and "14个", "142元" and "142 yuan"
4. Case insensitive: "Christmas" and "christmas"

If they are consistent, Judgement is 1; if they are different, Judgement is 0.
You must output exactly one character: either 1 or 0. Do not include any other text, punctuation, explanation, or leave the output empty.\n\n
"""
    example_1 = """
[Question]: What's the date of the first Monday after the game?
[Standard Answer]: December 22nd
[Model Answer]: December 22nd
Judgement:
1
"""
    demo_prompt = chat_template + example_1 + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model Answer]: {predicted_answer}
Judgement:"""
    return f"{demo_prompt}{test_prompt}"


def judge_answer_open_ended(predicted_answer, ground_truth, question, max_retries=3, base_delay=2):
    pred_clean = predicted_answer.strip().lower()
    gt_clean = ground_truth.strip().lower()
    if pred_clean == gt_clean:
        return 1.0, "exact_match"
    if NUMERIC_ONLY_PATTERN.fullmatch(pred_clean) and NUMERIC_ONLY_PATTERN.fullmatch(gt_clean):
        try:
            return (1.0, "numeric_match") if float(pred_clean) == float(gt_clean) else (0.0, "numeric_mismatch")
        except ValueError:
            pass
    pred_md, gt_md = _extract_month_day(predicted_answer), _extract_month_day(ground_truth)
    if pred_md and gt_md:
        return (1.0, "date_match") if pred_md == gt_md else (0.0, "date_mismatch")
    if not _judge_api_key:
        eval_logger.warning("No OPENAI_API_KEY/NVIDIA_API_KEY set; open-ended question scored 0.")
        return 0.0, "api_key_missing"

    full_prompt = _judge_chat_prompt(predicted_answer, ground_truth, question)
    for attempt in range(max_retries + 1):
        try:
            resp = _judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict binary judge. Read the provided question, standard answer, and model answer. If they express the same meaning, reply with the single digit 1. If they differ or you are unsure, reply with the single digit 0. Never leave the reply empty."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
                timeout=60,
            )
            response = (resp.choices[0].message.content or "").strip()
            if not response:
                if attempt == max_retries:
                    return 0.0, "empty_response"
                continue
            if "Judgement:" in response:
                response = response.split("Judgement:")[-1].strip()
            digits = "".join(ch for ch in response if ch in {"0", "1"})
            if digits:
                return (1.0, "model_judge") if digits[0] == "1" else (0.0, "model_judge")
            if attempt == max_retries:
                return 0.0, "max_retries_reached"
        except Exception as e:
            if attempt == max_retries:
                eval_logger.error(f"Judge API error: {e}")
                return 0.0, "api_error"
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
    return 0.0, "max_retries_reached"


def score_prediction(answer, ground_truth, category, format_type, question):
    """Dispatch to the official scoring function by category + answer format."""
    category = (category or "").lower()
    format_type = (format_type or "").lower()
    if category == "location identification":
        return verify_location(answer, ground_truth)
    if format_type == "multiple choice":
        return judge_answer_choice(answer, ground_truth)
    if format_type == "yes or no":
        return judge_answer_yes_no(answer, ground_truth)
    if format_type == "open-ended" and category in ["science", "daily commonsense reasoning"]:
        return judge_answer_open_ended(answer, ground_truth, question)
    # default fallback matches eval.py (treat unknown format as multiple choice)
    return judge_answer_choice(answer, ground_truth)


# ----------------------------------------------------------------------------
# lmms-eval hooks
# ----------------------------------------------------------------------------
def dailyclue_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def dailyclue_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    mode = (lmms_eval_specific_kwargs or {}).get("mode", "c")
    system = get_system_prompt(doc["category_1"], doc["format"], mode)
    user = build_user_prompt(doc["category_1"], doc["question"].strip(), mode)
    return f"{system}\n\n{user}"


def dailyclue_doc_to_target(doc):
    return doc["ground_truth"]


def dailyclue_process_results(doc, results):
    raw = results[0]
    answer = extract_answer_content(raw)
    if not answer:  # model didn't emit <answer> tags; fall back to the raw text
        answer = raw.strip()
    score, reason = score_prediction(answer, doc["ground_truth"], doc["category_1"], doc["format"], doc["question"])
    metric = CATEGORY_TO_METRIC.get(doc["category_1"], "other")
    entry = {
        "question_id": doc["question_id"],
        "score": score,
        "reason": reason,
        "category": doc["category_1"],
        "format": doc["format"],
    }
    return {metric: entry, "dailyclue_overall": entry}


def dailyclue_aggregate(results):
    if not results:
        return 0.0
    return 100.0 * sum(r["score"] for r in results) / len(results)


def dailyclue_aggregate_overall(results):
    """Overall micro-accuracy (%); logs per-category and per-format breakdowns."""
    by_cat, by_fmt = defaultdict(list), defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["score"])
        by_fmt[r["format"]].append(r["score"])
    for cat, s in sorted(by_cat.items()):
        eval_logger.info(f"DailyClue [{cat}]: {100.0 * sum(s) / len(s):.2f}% (n={len(s)})")
    for fmt, s in sorted(by_fmt.items()):
        eval_logger.info(f"DailyClue <{fmt}>: {100.0 * sum(s) / len(s):.2f}% (n={len(s)})")
    return 100.0 * sum(r["score"] for r in results) / len(results)
