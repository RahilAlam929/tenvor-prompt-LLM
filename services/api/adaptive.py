import re

QUESTION_BANK = {
    "ai": [
        ("goal", "What should the AI system actually do?"),
        ("users", "Who will use this system?"),
        ("data", "What data or knowledge will it need?"),
        ("compute", "What hardware or compute can you use?"),
        ("success", "How will you measure success?"),
    ],
    "software": [
        ("goal", "What problem should the software solve?"),
        ("users", "Who are the target users?"),
        ("platform", "Which platform: web, mobile, desktop or API?"),
        ("stack", "Any preferred programming language or tech stack?"),
        ("constraints", "What are your budget, deadline or security constraints?"),
    ],
    "business": [
        ("goal", "What business outcome do you want?"),
        ("audience", "Who is your target customer?"),
        ("budget", "What is your starting budget?"),
        ("channel", "How will you reach customers?"),
        ("success", "What result would make this successful?"),
    ],
    "writing": [
        ("goal", "What should this writing achieve?"),
        ("audience", "Who will read it?"),
        ("tone", "What tone should it use?"),
        ("length", "How long should it be?"),
        ("format", "What format do you need?"),
    ],
    "general": [
        ("goal", "What exact outcome do you want?"),
        ("audience", "Who is this for?"),
        ("constraints", "What limitations or requirements matter?"),
        ("format", "What output format do you prefer?"),
        ("success", "How will you judge the result?"),
    ],
}


def detect_domain(idea, requested="auto"):
    category = (requested or "auto").lower()

    aliases = {
        "ai": "ai",
        "ml": "ai",
        "llm": "ai",
        "machine learning": "ai",
        "software": "software",
        "coding": "software",
        "development": "software",
        "programming": "software",
        "business": "business",
        "startup": "business",
        "marketing": "business",
        "writing": "writing",
        "content": "writing",
        "copywriting": "writing",
    }

    if category not in ("auto", "general"):
        return aliases.get(category, "general")

    text = idea.lower()

    rules = {
        "ai": [
            "llm", "machine learning", "artificial intelligence",
            "model training", "neural network", " ai ",
        ],
        "software": [
            "app", "website", "software", "api", "code",
            "program", "backend", "frontend",
        ],
        "business": [
            "startup", "business", "customer", "revenue",
            "marketing", "sales", "product launch",
        ],
        "writing": [
            "write", "blog", "article", "story",
            "caption", "essay", "copywriting",
        ],
    }

    padded = " " + text + " "
    scores = {
        domain: sum(1 for word in words if word in padded)
        for domain, words in rules.items()
    }

    best = max(scores, key=scores.get)
    return best if scores[best] else "general"


def get_questions(idea, category="auto"):
    domain = detect_domain(idea, category)
    return {
        "domain": domain,
        "questions": [
            {"key": key, "question": question}
            for key, question in QUESTION_BANK[domain]
        ],
    }


def build_adaptive_prompt(idea, answers, category="auto", target="universal"):
    domain = detect_domain(idea, category)
    details = []

    for key, question in QUESTION_BANK[domain]:
        value = str(answers.get(key, "")).strip()
        if value:
            details.append(f"- {question}\n  Answer: {value}")

    context = "\n".join(details) or "No additional answers supplied."

    return f"""# ROLE
Act as an experienced specialist in {domain}.

# OBJECTIVE
{idea.strip()}

# USER REQUIREMENTS
{context}

# TARGET
{target}

# INSTRUCTIONS
1. Solve the stated objective using the supplied requirements.
2. Adapt the answer to the target audience and experience level.
3. Respect constraints, budget, platform and available tools.
4. Separate known facts from assumptions.
5. Ask follow-up questions only when essential information is missing.
6. Provide a practical, ordered solution.
7. Include examples and measurable success criteria where useful.

# OUTPUT
Choose a clear format appropriate to the task.

# FINAL CHECK
Verify that the response addresses the objective and the user's answers.
"""


def score_prompt(prompt):
    text = prompt.strip()
    words = re.findall(r"\b[\w'-]+\b", text)
    lower = text.lower()

    checks = {
        "objective": any(
            term in lower for term in ("objective", "goal", "task", "purpose")
        ),
        "instructions": any(
            term in lower for term in ("instructions", "requirements", "steps")
        ),
        "context": any(
            term in lower for term in ("context", "audience", "user requirements")
        ),
        "output_format": any(
            term in lower for term in ("output", "format", "deliverable")
        ),
        "success_criteria": any(
            term in lower for term in ("success", "criteria", "measure")
        ),
        "sufficient_detail": len(words) >= 50,
    }

    score = round(sum(checks.values()) / len(checks) * 100)

    return {
        "score": score,
        "checks": checks,
        "word_count": len(words),
        "note": (
            "Heuristic score, not an AI quality judgment. "
            "It measures structural checks, not actual answer quality."
        ),
    }
