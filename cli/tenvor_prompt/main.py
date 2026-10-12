from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="Tenvor Prompt — offline and AI prompt engineering")
console = Console()
API_URL = "http://127.0.0.1:8002"


def call_api(endpoint, payload):
    try:
        response = httpx.post(
            f"{API_URL}/v1/prompts/{endpoint}",
            json=payload,
            timeout=240.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        console.print(f"[red]API error:[/red] {detail}")
        raise typer.Exit(code=1) from exc
    except httpx.RequestError as exc:
        console.print(
            "[red]API unavailable.[/red] Start it with:\n"
            "uvicorn services.api.main:app --reload --port 8002"
        )
        raise typer.Exit(code=1) from exc


def display_result(data, output=""):
    result = data["result"]
    console.print(
        Panel(
            result,
            title=f"Tenvor Prompt | {data['provider']}",
            expand=False,
        )
    )
    if output:
        Path(output).write_text(result + "\n", encoding="utf-8")
        console.print(f"[green]Saved to {output}[/green]")


@app.command()
def interview(
    idea: str = typer.Argument(..., help="Describe what you want to achieve"),
    output: str = typer.Option("final-prompt.md", help="Output Markdown file"),
):
    console.print(
        Panel(
            "I'll ask a few questions to personalize your prompt.\n"
            "Press Enter to skip any optional question.",
            title="Tenvor Prompt — Interview Mode",
        )
    )

    category = typer.prompt("Domain (AI, business, education, writing, etc.)",
                            default="general")
    audience = typer.prompt("Who is this for?", default="general users")
    goal = typer.prompt("What outcome do you want?", default=idea)
    constraints = typer.prompt(
        "Any budget, tools, deadline or other constraints?",
        default="No specific constraints provided",
    )
    format_choice = typer.prompt(
        "Preferred output (plan, code, report, steps, etc.)",
        default="step-by-step plan",
    )
    experience = typer.prompt(
        "Your experience level (beginner/intermediate/advanced)",
        default="beginner",
    )

    context = (
        f"Audience: {audience}\n"
        f"Desired outcome: {goal}\n"
        f"Constraints: {constraints}\n"
        f"Preferred output: {format_choice}\n"
        f"Experience level: {experience}"
    )

    data = call_api(
        "generate",
        {
            "idea": idea,
            "category": category,
            "target": "universal",
            "context": context,
            "provider": "rules",
        },
    )
    display_result(data, output)


@app.command()
def generate(
    idea: str = typer.Argument(..., help="Your task or idea"),
    category: str = typer.Option("auto", help="Task domain"),
    target: str = typer.Option("universal", help="Target AI model/tool"),
    context: str = typer.Option("", help="Additional context"),
    provider: str = typer.Option("rules", help="rules, ollama or openai"),
    model: str = typer.Option("", help="Optional model override"),
    output: str = typer.Option("", help="Save result to a file"),
):
    data = call_api(
        "generate",
        {
            "idea": idea,
            "category": category,
            "target": target,
            "context": context,
            "provider": provider,
            "model": model,
        },
    )
    display_result(data, output)


@app.command()
def optimize(
    prompt: Optional[str] = typer.Argument(None, help="Existing prompt text"),
    file: str = typer.Option("", "--file", help="Read prompt from a file"),
    goal: str = typer.Option(
        "Improve clarity, completeness and usefulness",
        help="Optimization goal",
    ),
    target: str = typer.Option("universal", help="Target model/tool"),
    provider: str = typer.Option("rules", help="rules, ollama or openai"),
    model: str = typer.Option("", help="Optional model override"),
    output: str = typer.Option("", help="Save optimized prompt to a file"),
):
    if file:
        try:
            prompt_text = Path(file).read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Cannot read file:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    elif prompt:
        prompt_text = prompt
    else:
        console.print("Provide prompt text or use --file prompt.md")
        raise typer.Exit(code=2)

    data = call_api(
        "optimize",
        {
            "prompt": prompt_text,
            "goal": goal,
            "target": target,
            "provider": provider,
            "model": model,
        },
    )
    display_result(data, output)


@app.command()
def health():
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        console.print_json(response.text)
    except httpx.HTTPError as exc:
        console.print("[red]API is not running.[/red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()


@app.command()
def adaptive(
    idea: str = typer.Argument(..., help="Describe your goal"),
    category: str = typer.Option("auto", help="Domain"),
    target: str = typer.Option("universal", help="Target AI model/tool"),
    output: str = typer.Option("final-prompt.md", help="Output file"),
):
    questions = call_api(
        "unused",
        {},
    ) if False else None

    try:
        response = httpx.post(
            f"{API_URL}/v1/interview/questions",
            json={"idea": idea, "category": category},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        console.print("[red]Could not retrieve interview questions.[/red]")
        raise typer.Exit(code=1) from exc

    answers = {}
    console.print(Panel(
        f"Goal: {idea}\nDomain detected: {data['domain']}",
        title="Tenvor Adaptive Interview",
    ))

    for item in data["questions"]:
        answer = typer.prompt(item["question"], default="", show_default=False)
        if answer.strip():
            answers[item["key"]] = answer.strip()

    result = call_api(
        "unused",
        {},
    ) if False else None

    try:
        response = httpx.post(
            f"{API_URL}/v1/interview/build",
            json={
                "idea": idea,
                "category": category,
                "target": target,
                "answers": answers,
            },
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as exc:
        console.print("[red]Could not build the prompt.[/red]")
        raise typer.Exit(code=1) from exc

    display_result(result, output)


@app.command()
def score(
    file: str = typer.Argument(..., help="Markdown or text prompt file"),
):
    try:
        prompt_text = Path(file).read_text(encoding="utf-8")
        response = httpx.post(
            f"{API_URL}/v1/prompts/score",
            json={"prompt": prompt_text},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (OSError, httpx.HTTPError) as exc:
        console.print(f"[red]Scoring failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(Panel(
        f"Score: {data['score']}/100\n"
        f"Words: {data['word_count']}\n\n"
        + "\n".join(
            f"{'✓' if passed else '✗'} {name.replace('_', ' ').title()}"
            for name, passed in data["checks"].items()
        )
        + f"\n\n{data['note']}",
        title="Prompt Quality Check",
    ))
