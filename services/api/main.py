from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.api.llm import LLMError, call_llm

app = FastAPI(title="Tenvor Prompt", version="0.3.0")


class GenerateRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=10000)
    category: str = "auto"
    target: str = "universal"
    context: str = Field(default="", max_length=10000)
    provider: str = "rules"
    model: str = ""


class OptimizeRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=20000)
    goal: str = "Improve clarity and usefulness"
    target: str = "universal"
    provider: str = "rules"
    model: str = ""


class AIResponse(BaseModel):
    result: str
    provider: str
    model: str


def offline_generate(request):
    return f"""# ROLE
Act as a specialist in {request.category}.

# OBJECTIVE
{request.idea}

# TARGET
Prepare the output for: {request.target}

# USER CONTEXT
{request.context.strip() or "No additional context supplied."}

# INSTRUCTIONS
1. Break the objective into clear, actionable steps.
2. Respect the user's stated goals and constraints.
3. Identify missing information instead of inventing facts.
4. Explain important decisions in understandable language.
5. Choose tools and methods appropriate to this task.
6. Include examples where they help the user.
7. State assumptions and limitations explicitly.

# OUTPUT FORMAT
Start with a brief summary, then provide the solution in an
organized format appropriate to the task.

# SUCCESS CRITERIA
The result should directly address the objective, respect the
context, and include practical next steps.
"""


def offline_optimize(request):
    return f"""# OPTIMIZED PROMPT

## Role
Act as a qualified specialist who can complete the requested task.

## Objective
{request.prompt.strip()}

## Optimization goal
{request.goal}

## Instructions
1. Preserve the original intent.
2. Make instructions specific and unambiguous.
3. State relevant constraints and assumptions.
4. Ask questions if essential details are missing.
5. Specify the expected output format.
6. Avoid unsupported claims and unnecessary repetition.
7. Define what a successful result looks like.

## Target
{request.target}

## Optimization notes
This offline version organizes the prompt into a clearer structure.
Review the objective and add any missing task-specific details.
"""


def run_ai(provider, model, system, user):
    try:
        result = call_llm(provider, system, user, model or None)
        return AIResponse(
            result=result["text"],
            provider=result["provider"],
            model=result["model"],
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/")
def root():
    return {"service": "Tenvor Prompt", "version": "0.3.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "tenvor-prompt", "version": "0.3.0"}


@app.post("/v1/prompts/generate", response_model=AIResponse)
def generate_prompt(request: GenerateRequest):
    if request.provider.lower() == "rules":
        return AIResponse(
            result=offline_generate(request),
            provider="rules",
            model="adaptive-template-v1",
        )

    system = """
You are Tenvor Prompt, a universal prompt engineer.
Create a specific, actionable prompt for the user's objective.
Include relevant context, constraints, output format and success criteria.
Do not perform the task itself. Return the finished prompt only.
"""
    user = (
        f"OBJECTIVE: {request.idea}\n"
        f"DOMAIN: {request.category}\n"
        f"TARGET: {request.target}\n"
        f"CONTEXT: {request.context or 'Not provided'}"
    )
    return run_ai(request.provider, request.model, system, user)


@app.post("/v1/prompts/optimize", response_model=AIResponse)
def optimize_prompt(request: OptimizeRequest):
    if request.provider.lower() == "rules":
        return AIResponse(
            result=offline_optimize(request),
            provider="rules",
            model="offline-optimizer-v1",
        )

    system = """
You are an expert prompt optimizer.
Preserve the original intention, improve clarity and completeness,
and include useful constraints and output criteria.
Return the optimized prompt followed by brief optimization notes.
"""
    user = (
        f"PROMPT:\n{request.prompt}\n\n"
        f"GOAL: {request.goal}\nTARGET: {request.target}"
    )
    return run_ai(request.provider, request.model, system, user)


from services.api.adaptive import (
    build_adaptive_prompt,
    get_questions,
    score_prompt,
)


class QuestionsRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=10000)
    category: str = "auto"


class AdaptiveBuildRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=10000)
    category: str = "auto"
    target: str = "universal"
    answers: dict = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)


@app.post("/v1/interview/questions")
def interview_questions(request: QuestionsRequest):
    return get_questions(request.idea, request.category)


@app.post("/v1/interview/build")
def interview_build(request: AdaptiveBuildRequest):
    prompt = build_adaptive_prompt(
        request.idea,
        request.answers,
        request.category,
        request.target,
    )
    return {
        "result": prompt,
        "provider": "rules",
        "model": "adaptive-question-engine-v1",
    }


@app.post("/v1/prompts/score")
def prompt_score(request: ScoreRequest):
    return score_prompt(request.prompt)
