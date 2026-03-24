import json
from pathlib import Path
from typing import Optional

from openai import OpenAI

_DEFAULT_SECRETS_PATH = Path(__file__).resolve().parent / "llm_secrets.json"


def _load_llm_secrets(path: Optional[Path] = None) -> dict:
    p = path or _DEFAULT_SECRETS_PATH
    p = p.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}; create it with "
            '{"api_key": "<key>", "base_url": "http://host:port/v1/"}'
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def LLM_demo(
    human_input,
    model_name="gpt-4o-mini",
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
    secrets_path: Optional[Path] = None,
):
    data = _load_llm_secrets(secrets_path)
    api_key = data.get("api_key")
    base_url = data.get("base_url")
    if not api_key:
        raise ValueError('llm_secrets.json must contain "api_key"')
    if not base_url:
        raise ValueError('llm_secrets.json must contain "base_url"')

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        # 调用 Chat Completion API 并设置参数
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": human_input},
            ],
            max_tokens=max_tokens,
        )
        return completion.model_dump()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    model_name = "Qwen2.5-0.5b"
    response1 = LLM_demo("你好，你是谁？", model_name)
    print(response1)
    response2 = LLM_demo("Who are you?", model_name)
    print(response2)
