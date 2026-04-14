import json
import re
from utils.ollama_client import query_model

# ─────────────────────────────────────────────
#  VERIFY FUNCTIONS — pure Python, no ML
# ─────────────────────────────────────────────

def verify_valid_json_with_keys(response: str) -> tuple[bool, str]:
    """Check if response is valid JSON containing 'name' and 'age' keys."""
    try:
        cleaned = response.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)
        if "name" in data and "age" in data:
            return True, "✅ Valid JSON with correct keys"
        else:
            missing = [k for k in ["name", "age"] if k not in data]
            return False, f"❌ Missing keys: {missing}"
    except json.JSONDecodeError:
        return False, f"❌ Not valid JSON — got: {response[:80]}"

def verify_sentence_count(response: str, expected: int) -> tuple[bool, str]:
    """Count sentences by splitting on . ! ? endings."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    sentences = [s for s in sentences if len(s.strip()) > 3]  # filter empty/tiny
    actual = len(sentences)
    if actual == expected:
        return True, f"✅ Exactly {actual} sentences"
    return False, f"❌ Expected {expected} sentences, got {actual}"

def verify_numbered_list_count(response: str, expected: int) -> tuple[bool, str]:
    """Count numbered list items (1. 2. 3. format)."""
    items = re.findall(r'^\s*\d+[\.\)]\s+.+', response, re.MULTILINE)
    actual = len(items)
    if actual == expected:
        return True, f"✅ Exactly {actual} numbered items"
    return False, f"❌ Expected {expected} items, got {actual}"

def verify_uppercase_only(response: str) -> tuple[bool, str]:
    """Check if entire response is uppercase (ignoring numbers/punctuation)."""
    letters = [c for c in response if c.isalpha()]
    if not letters:
        return False, "❌ No alphabetic characters found"
    all_upper = all(c.isupper() for c in letters)
    if all_upper:
        return True, "✅ All uppercase"
    lower_chars = [c for c in letters if c.islower()]
    return False, f"❌ Found {len(lower_chars)} lowercase letters"

def verify_word_not_present(response: str, forbidden_word: str) -> tuple[bool, str]:
    """Check that a specific word does not appear in the response."""
    pattern = r'\b' + re.escape(forbidden_word.lower()) + r'\b'
    found = re.search(pattern, response.lower())
    if not found:
        return True, f"✅ Word '{forbidden_word}' not used"
    return False, f"❌ Forbidden word '{forbidden_word}' found in response"

def verify_single_word(response: str) -> tuple[bool, str]:
    """Check if response is exactly one word."""
    words = response.strip().split()
    words = [w for w in words if w.strip(".,!?;:")]
    if len(words) == 1:
        return True, "✅ Single word response"
    return False, f"❌ Expected 1 word, got {len(words)}: '{response[:60]}'"

def verify_yes_or_no(response: str) -> tuple[bool, str]:
    """Check if response is only Yes or No (with optional punctuation)."""
    cleaned = response.strip().lower().strip(".,!?")
    if cleaned in ["yes", "no"]:
        return True, f"✅ Valid yes/no answer"
    return False, f"❌ Not a yes/no answer: '{response[:60]}'"

def verify_starts_with(response: str, prefix: str) -> tuple[bool, str]:
    """Check if response starts with a specific word/phrase."""
    if response.strip().lower().startswith(prefix.lower()):
        return True, f"✅ Starts with '{prefix}'"
    return False, f"❌ Does not start with '{prefix}': '{response[:60]}'"

def verify_numeric_only(response: str) -> tuple[bool, str]:
    """Check if response is just a number."""
    cleaned = response.strip().replace(",", "")
    try:
        float(cleaned)
        return True, "✅ Numeric only response"
    except ValueError:
        return False, f"❌ Not a number: '{response[:60]}'"

def verify_paragraph_count(response: str, expected: int) -> tuple[bool, str]:
    """Count paragraphs separated by blank lines."""
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    actual = len(paragraphs)
    if actual == expected:
        return True, f"✅ Exactly {actual} paragraphs"
    return False, f"❌ Expected {expected} paragraphs, got {actual}"


# ─────────────────────────────────────────────
#  TEST CASES
# ─────────────────────────────────────────────

TEST_CASES = [
    {
        "id": 1,
        "description": "JSON format with specific keys",
        "prompt": 'Respond with ONLY a raw JSON object containing exactly two keys: "name" (any string) and "age" (any integer). No explanation, no markdown code blocks, no extra text. Just the JSON object itself.',
        "verify": lambda r: verify_valid_json_with_keys(r),
        "category": "Format Compliance"
    },
    {
        "id": 2,
        "description": "Exactly 3 sentences",
        "prompt": "Write exactly 3 sentences explaining why Python is popular. Not 2, not 4. Exactly 3 sentences.",
        "verify": lambda r: verify_sentence_count(r, 3),
        "category": "Length Control"
    },
    {
        "id": 3,
        "description": "Numbered list of exactly 5 items",
        "prompt": "List exactly 5 benefits of exercise. Use numbered format (1. 2. 3. 4. 5.). No introduction, no conclusion. Just the 5 numbered items.",
        "verify": lambda r: verify_numbered_list_count(r, 5),
        "category": "Format Compliance"
    },
    {
        "id": 4,
        "description": "All uppercase response",
        "prompt": "What is the capital of Japan? Respond in UPPERCASE LETTERS ONLY. Every single letter must be uppercase.",
        "verify": lambda r: verify_uppercase_only(r),
        "category": "Format Compliance"
    },
    {
        "id": 5,
        "description": "Avoid forbidden word 'the'",
        "prompt": "Explain what a database is in 2-3 sentences. Important constraint: do NOT use the word 'the' anywhere in your response. Find alternative phrasing.",
        "verify": lambda r: verify_word_not_present(r, "the"),
        "category": "Constraint Following"
    },
    {
        "id": 6,
        "description": "Single word response",
        "prompt": "What programming language is used for data science most often? Respond with only ONE single word. Nothing else.",
        "verify": lambda r: verify_single_word(r),
        "category": "Length Control"
    },
    {
        "id": 7,
        "description": "Yes or No only",
        "prompt": "Is Python an interpreted language? Answer with only Yes or No. A single word. No explanation.",
        "verify": lambda r: verify_yes_or_no(r),
        "category": "Constraint Following"
    },
    {
        "id": 8,
        "description": "Response must start with 'Answer:'",
        "prompt": "What is 15 multiplied by 7? You must begin your response with exactly the word 'Answer:' followed by your response.",
        "verify": lambda r: verify_starts_with(r, "Answer:"),
        "category": "Format Compliance"
    },
    {
        "id": 9,
        "description": "Numeric only response",
        "prompt": "How many days are in a leap year? Respond with only the number. No words, no units, just the number.",
        "verify": lambda r: verify_numeric_only(r),
        "category": "Format Compliance"
    },
    {
        "id": 10,
        "description": "Exactly 2 paragraphs",
        "prompt": "Write about the advantages and disadvantages of remote work. Structure your response in exactly 2 paragraphs separated by a blank line. First paragraph: advantages. Second paragraph: disadvantages.",
        "verify": lambda r: verify_paragraph_count(r, 2),
        "category": "Structure Compliance"
    },
    {
        "id": 11,
        "description": "Exactly 2 sentences",
        "prompt": "Describe machine learning in exactly 2 sentences. No more, no less.",
        "verify": lambda r: verify_sentence_count(r, 2),
        "category": "Length Control"
    },
    {
        "id": 12,
        "description": "Numbered list of exactly 3 items",
        "prompt": "Give me exactly 3 reasons why sleep is important. Numbered list (1. 2. 3.) only. No intro or outro text.",
        "verify": lambda r: verify_numbered_list_count(r, 3),
        "category": "Format Compliance"
    },
]


# ─────────────────────────────────────────────
#  MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────

def run_instruction_following(model_name: str, sample_size: int = None, progress_callback=None):
    """
    Run instruction following evaluation on a model.
    Returns (overall_score, detailed_results_list)
    """
    tests = TEST_CASES if sample_size is None else TEST_CASES[:sample_size]
    results = []
    passed = 0

    for i, test in enumerate(tests):
        response, latency = query_model(model_name, test["prompt"])

        if response == "":
            success = False
            feedback = "❌ Model returned empty response"
        else:
            success, feedback = test["verify"](response)

        if success:
            passed += 1

        results.append({
            "id": test["id"],
            "description": test["description"],
            "category": test["category"],
            "prompt": test["prompt"],
            "response": response,
            "passed": success,
            "feedback": feedback,
            "latency_s": latency
        })

        if progress_callback:
            progress_callback(i + 1, len(tests), test["description"])

    overall_score = round((passed / len(tests)) * 100, 1)
    return overall_score, results