import os

import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    pass


_tenvor_instance = None


def get_tenvor():
    global _tenvor_instance

    if _tenvor_instance is None:
        from training.src.inference import TenvorInference
        _tenvor_instance = TenvorInference()

    return _tenvor_instance


def call_llm(provider, system_prompt, user_prompt, model=None):
    provider = provider.lower().strip()

    try:
        if provider == "tenvor":
            try:
                engine = get_tenvor()
                generated = engine.generate(
                    user_prompt,
                    max_new_tokens=120,
                )
                content = generated[len(user_prompt):].strip()
                model = model or "tenvor-gpt"
            except ValueError as exc:
                raise LLMError(
                    "TenvorGPT cannot process some characters in this prompt. "
                    "Its character vocabulary is limited by the training data."
                ) from exc

        elif provider == "ollama":
            base_url = os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/")
            model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")

            response = httpx.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=180.0,
            )
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "")

        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise LLMError(
                    "OPENAI_API_KEY missing. Add it to your .env file."
                )

            base_url = os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/")
            model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            response = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.4,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

        else:
            raise LLMError(
                "Unsupported provider. Choose 'tenvor', 'ollama', or 'openai'."
            )

        if not isinstance(content, str) or not content.strip():
            raise LLMError("The model returned an empty response.")

        return {"text": content.strip(), "provider": provider, "model": model}

    except LLMError:
        raise
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise LLMError(
            f"Provider returned HTTP {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.RequestError as exc:
        raise LLMError(
            "Cannot connect to the selected AI provider. "
            "Check its URL and whether the service is running."
        ) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(
            "Unexpected response from the AI provider."
        ) from exc
