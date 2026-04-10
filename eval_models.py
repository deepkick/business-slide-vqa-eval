"""各VLMモデルへのAPI呼び出しとLLM-as-a-Judgeスコアリング."""

import base64
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "あなたは日本語のビジネス資料を読み解くAIアシスタントです。"
    "画像を注意深く観察し、質問に対して正確かつ簡潔に日本語で回答してください。"
)

JUDGE_PROMPT_TEMPLATE = """以下の質問に対する正解とモデルの回答を比較し、1〜5のスコアで評価してください。
1: 完全に不正解 2: 一部正しいが重要な誤りあり 3: おおむね正しいが不完全
4: ほぼ正解 5: 完全に正解
数字のみ回答してください。

質問: {question}
正解: {answer}
モデルの回答: {model_answer}"""

# レート制限対策
PRE_CALL_DELAY = 2.0  # 各API呼び出し前のスリープ秒数
RETRY_BACKOFFS = [5, 10, 20]  # 429時のリトライ間隔（最大3リトライ）


def _is_rate_limit_error(exc: Exception) -> bool:
    """例外が429 (rate limit) かどうかを判定."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _call_with_retry(fn):
    """API呼び出しをリトライ付きで実行する.

    各呼び出し前に PRE_CALL_DELAY 秒スリープし、
    429エラー時は RETRY_BACKOFFS 秒待ってリトライする。
    """
    last_exc: Exception | None = None
    for attempt in range(len(RETRY_BACKOFFS) + 1):
        time.sleep(PRE_CALL_DELAY)
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_rate_limit_error(e) or attempt >= len(RETRY_BACKOFFS):
                raise
            time.sleep(RETRY_BACKOFFS[attempt])
    # 到達不能だが念のため
    assert last_exc is not None
    raise last_exc


def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_media_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower().lstrip(".")
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix, "image/png"
    )


def get_available_models() -> dict[str, bool]:
    """各モデルのAPIキーが設定されているかチェック."""
    return {
        "Claude Sonnet 4.6": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "GPT-4.1": bool(os.environ.get("OPENAI_API_KEY")),
        "Qwen2.5-VL-72B": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


def ask_claude(image_path: str | Path, question: str) -> tuple[str, float]:
    """Claude Sonnet 4.6に質問し、(回答, 応答時間秒)を返す."""
    import anthropic

    client = anthropic.Anthropic(timeout=60.0)
    b64 = _encode_image(image_path)
    media_type = _get_media_type(image_path)

    start = time.time()
    try:
        response = _call_with_retry(
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            )
        )
        elapsed = time.time() - start
        return response.content[0].text, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"エラー: {e}", elapsed


def ask_gpt4(image_path: str | Path, question: str) -> tuple[str, float]:
    """GPT-4.1に質問し、(回答, 応答時間秒)を返す."""
    from openai import OpenAI

    client = OpenAI(timeout=60.0)
    b64 = _encode_image(image_path)
    media_type = _get_media_type(image_path)

    start = time.time()
    try:
        response = _call_with_retry(
            lambda: client.chat.completions.create(
                model="gpt-4.1",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}",
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ],
            )
        )
        elapsed = time.time() - start
        return response.choices[0].message.content, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"エラー: {e}", elapsed


def ask_qwen(image_path: str | Path, question: str) -> tuple[str, float]:
    """Qwen2.5-VL-72Bに質問し、(回答, 応答時間秒)を返す（OpenRouter経由）."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )
    b64 = _encode_image(image_path)
    media_type = _get_media_type(image_path)

    start = time.time()
    try:
        response = _call_with_retry(
            lambda: client.chat.completions.create(
                model="qwen/qwen2.5-vl-72b-instruct",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}",
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ],
            )
        )
        elapsed = time.time() - start
        return response.choices[0].message.content, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"エラー: {e}", elapsed


MODEL_FUNCTIONS = {
    "Claude Sonnet 4.6": ask_claude,
    "GPT-4.1": ask_gpt4,
    "Qwen2.5-VL-72B": ask_qwen,
}


def judge_answer(question: str, answer: str, model_answer: str) -> int:
    """Claude Sonnet 4.6をジャッジとして使い、1-5のスコアを返す."""
    import anthropic

    client = anthropic.Anthropic(timeout=60.0)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, answer=answer, model_answer=model_answer
    )

    try:
        response = _call_with_retry(
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        text = response.content[0].text.strip()
        match = re.search(r"[1-5]", text)
        return int(match.group()) if match else 0
    except Exception:
        return 0
