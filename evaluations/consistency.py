import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.ollama_client import query_model

# Load once at module level — avoids reloading on every call
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


# ─────────────────────────────────────────────
#  QUESTION BANK
#  Each entry has a base question + 4 paraphrases = 5 total phrasings
#  All 5 should produce semantically similar answers from a consistent model
# ─────────────────────────────────────────────

QUESTION_BANK = [
    {
        "id": 1,
        "topic": "Python use case",
        "phrasings": [
            "What is Python most commonly used for?",
            "What are the main use cases of Python?",
            "In what fields is Python most popular?",
            "Name the primary applications of the Python language.",
            "What kinds of tasks do developers use Python for most often?"
        ]
    },
    {
        "id": 2,
        "topic": "Definition of machine learning",
        "phrasings": [
            "What is machine learning?",
            "Can you explain machine learning in simple terms?",
            "Define machine learning.",
            "How would you describe what machine learning is?",
            "What does the term machine learning mean?"
        ]
    },
    {
        "id": 3,
        "topic": "HTTP vs HTTPS",
        "phrasings": [
            "What is the difference between HTTP and HTTPS?",
            "How does HTTPS differ from HTTP?",
            "Why is HTTPS more secure than HTTP?",
            "What makes HTTPS different from HTTP?",
            "Explain the distinction between HTTP and HTTPS."
        ]
    },
    {
        "id": 4,
        "topic": "RAM purpose",
        "phrasings": [
            "What does RAM do in a computer?",
            "What is the purpose of RAM?",
            "Why do computers need RAM?",
            "What role does RAM play in a computer system?",
            "Explain what RAM is used for."
        ]
    },
    {
        "id": 5,
        "topic": "API definition",
        "phrasings": [
            "What is an API?",
            "Can you explain what an API is?",
            "What does API stand for and what does it do?",
            "Define what an API is in software development.",
            "How would you explain an API to someone new to programming?"
        ]
    },
    {
        "id": 6,
        "topic": "Git purpose",
        "phrasings": [
            "What is Git used for?",
            "Why do developers use Git?",
            "What problem does Git solve?",
            "Explain the purpose of Git.",
            "What is Git and why is it important?"
        ]
    },
    {
        "id": 7,
        "topic": "Cloud computing",
        "phrasings": [
            "What is cloud computing?",
            "How would you explain cloud computing?",
            "Define cloud computing.",
            "What does cloud computing mean?",
            "Can you describe what cloud computing is?"
        ]
    },
    {
        "id": 8,
        "topic": "Neural network basics",
        "phrasings": [
            "What is a neural network?",
            "How would you explain a neural network?",
            "What does a neural network do?",
            "Can you define what a neural network is?",
            "Describe how a neural network works."
        ]
    },
]


# ─────────────────────────────────────────────
#  CORE SCORING LOGIC
# ─────────────────────────────────────────────

def score_consistency(responses: list[str]) -> tuple[float, np.ndarray]:
    """
    Given a list of responses to paraphrased versions of the same question,
    embed all responses and compute the average pairwise cosine similarity.

    Returns:
        avg_similarity (float): 0.0 to 1.0 — higher means more consistent
        similarity_matrix (np.ndarray): full pairwise matrix for debugging
    """
    embedder = get_embedder()

    # Filter out empty responses
    valid = [r for r in responses if r.strip()]
    if len(valid) < 2:
        return 0.0, np.array([])

    embeddings = embedder.encode(valid)
    sim_matrix = cosine_similarity(embeddings)

    # Average of upper triangle (exclude diagonal self-similarity = 1.0)
    n = len(valid)
    upper_indices = np.triu_indices(n, k=1)
    avg_sim = float(np.mean(sim_matrix[upper_indices]))

    return round(avg_sim, 4), sim_matrix


def interpret_score(score: float) -> tuple[str, str]:
    """Return a human-readable label and color hint for a consistency score."""
    if score >= 0.90:
        return "Highly Consistent", "✅"
    elif score >= 0.75:
        return "Mostly Consistent", "🟡"
    elif score >= 0.60:
        return "Somewhat Inconsistent", "🟠"
    else:
        return "Inconsistent", "🔴"


# ─────────────────────────────────────────────
#  MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────

def run_consistency(model_name: str, sample_size: int = None, progress_callback=None):
    """
    Run consistency evaluation on a model.

    For each question topic, send all 5 paraphrased versions to the model,
    embed the responses, and compute average pairwise cosine similarity.

    Returns:
        overall_score (float): 0–100 scaled average consistency across all topics
        detailed_results (list): per-topic breakdown with responses and scores
    """
    questions = QUESTION_BANK if sample_size is None else QUESTION_BANK[:sample_size]
    results = []
    total_similarity = 0.0

    for i, question in enumerate(questions):
        topic_responses = []
        topic_latencies = []

        # Query model with all 5 phrasings
        for phrasing in question["phrasings"]:
            response, latency = query_model(
                model_name,
                phrasing,
                system_prompt="Answer concisely in 2-3 sentences."
            )
            topic_responses.append(response)
            topic_latencies.append(latency)

        # Score consistency across the 5 responses
        avg_sim, sim_matrix = score_consistency(topic_responses)
        label, emoji = interpret_score(avg_sim)
        total_similarity += avg_sim

        results.append({
            "id": question["id"],
            "topic": question["topic"],
            "phrasings": question["phrasings"],
            "responses": topic_responses,
            "latencies": topic_latencies,
            "consistency_score": avg_sim,
            "consistency_pct": round(avg_sim * 100, 1),
            "label": label,
            "emoji": emoji,
            "avg_latency": round(sum(topic_latencies) / len(topic_latencies), 2)
        })

        if progress_callback:
            progress_callback(i + 1, len(questions), question["topic"])

    overall_score = round((total_similarity / len(questions)) * 100, 1)
    return overall_score, results