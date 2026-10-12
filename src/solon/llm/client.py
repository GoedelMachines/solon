"""LLM client: unified interface for multiple providers."""

import os
from dotenv import load_dotenv

load_dotenv()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "anthropic",
    model: str = None,
    reasoning_effort: str = None,
    max_tokens: int = 16000,
) -> str:
    """Call an LLM. Returns the raw text response.

    Supports: anthropic, openai, gemini, deepseek, together, openrouter.
    """
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model or "claude-opus-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        model = model or "gpt-5.2"
        kwargs = {
            "model": model,
            "messages": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    elif provider == "gemini":
        import openai
        client = openai.OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        response = client.chat.completions.create(
            model=model or "gemini-3-pro-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    elif provider == "deepseek":
        import openai
        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model=model or "deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    elif provider == "together":
        import openai
        client = openai.OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )
        response = client.chat.completions.create(
            model=model or "deepseek-ai/DeepSeek-V3.1",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    elif provider == "openrouter":
        import openai
        client = openai.OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=model or "deepseek/deepseek-v3.2",
            max_tokens=max(max_tokens, 65536),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: anthropic, openai, gemini, deepseek, together, openrouter"
        )
