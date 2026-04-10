"""BusinessSlideVQA Evaluator - フロンティアVLM比較 Gradio アプリ."""

import json
from pathlib import Path

import gradio as gr

from eval_models import MODEL_FUNCTIONS, get_available_models, judge_answer

DATA_DIR = Path(__file__).resolve().parent / "data"
SUBSET_JSON = DATA_DIR / "vqa_subset.json"
IMAGES_DIR = DATA_DIR / "images"


def load_dataset() -> list[dict]:
    with open(SUBSET_JSON, encoding="utf-8") as f:
        return json.load(f)


DATASET = load_dataset()
IMAGE_TYPES = sorted(set(q["image type"] for q in DATASET))

# 白文字が読めるよう、彩度を上げ濃い色に統一
SCORE_COLORS = {
    5: "#16a34a",  # 濃緑
    4: "#22c55e",  # 緑
    3: "#d97706",  # 濃黄/琥珀
    2: "#ea580c",  # 濃橙
    1: "#dc2626",  # 濃赤
    0: "#6b7280",  # グレー
}

# ダークテーマ用カラーパレット
COLOR_BORDER = "#4b5563"
COLOR_HEADER_BG = "#374151"
COLOR_HEADER_FG = "#f3f4f6"
COLOR_CELL_FG = "#e5e7eb"
COLOR_MUTED = "#9ca3af"
COLOR_LINK = "#60a5fa"

# Gradio全体のCSS（リンク色など）
CUSTOM_CSS = f"""
a, .markdown a, .prose a {{
    color: {COLOR_LINK} !important;
    text-decoration: underline;
}}
a:hover {{
    color: #93c5fd !important;
}}
"""


def get_filtered_questions(image_type: str) -> gr.Dropdown:
    items = DATASET if image_type == "全て" else [q for q in DATASET if q["image type"] == image_type]
    choices = [f"Q{q['question_id']}: {q['question'][:20]}..." for q in items]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def get_question_by_label(label: str) -> dict | None:
    qid = int(label.split(":")[0].replace("Q", ""))
    return next((q for q in DATASET if q["question_id"] == qid), None)


# ---------- タブ1: 1問ずつ評価 ----------


def run_single_eval(image_type: str, question_label: str, selected_models: list[str]):
    if not question_label:
        return None, "", "", ""

    q = get_question_by_label(question_label)
    if q is None:
        return None, "", "", ""

    image_path = IMAGES_DIR / q["image"]
    question_text = q["question"]
    answer_text = q["answer"]

    results_html_parts = []
    for model_name in selected_models:
        func = MODEL_FUNCTIONS.get(model_name)
        if func is None:
            continue

        model_answer, elapsed = func(image_path, question_text)
        score = judge_answer(question_text, answer_text, model_answer)
        color = SCORE_COLORS.get(score, "#9ca3af")

        results_html_parts.append(f"""
        <div style="border:1px solid {COLOR_BORDER}; border-radius:8px; padding:16px; margin-bottom:12px; color:{COLOR_CELL_FG};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h3 style="margin:0; font-size:1.1em; color:{COLOR_HEADER_FG};">{model_name}</h3>
                <div style="display:flex; gap:12px; align-items:center;">
                    <span style="background:{color}; color:#ffffff; padding:4px 14px; border-radius:12px; font-weight:bold; font-size:0.95em;">
                        スコア: {score}/5
                    </span>
                    <span style="color:{COLOR_MUTED}; font-size:0.9em;">{elapsed:.1f}秒</span>
                </div>
            </div>
            <p style="margin:0; white-space:pre-wrap; color:{COLOR_CELL_FG};">{model_answer}</p>
        </div>
        """)

    results_html = "".join(results_html_parts) if results_html_parts else "<p>モデルを選択してください。</p>"

    return str(image_path), question_text, answer_text, results_html


# ---------- タブ2: バッチ評価 ----------


def run_batch_eval(image_type: str, selected_models: list[str], progress=gr.Progress()):
    items = DATASET if image_type == "全て" else [q for q in DATASET if q["image type"] == image_type]

    if not selected_models:
        return "<p>モデルを選択してください。</p>", "<p></p>"

    rows = []
    total = len(items)

    for i, q in enumerate(items):
        progress((i + 1) / total, desc=f"評価中... {i + 1}/{total}")
        image_path = IMAGES_DIR / q["image"]
        row = {
            "question_id": q["question_id"],
            "画像タイプ": q["image type"],
            "質問": q["question"][:30] + "...",
            "正解": q["answer"][:30] + "...",
        }

        for model_name in selected_models:
            func = MODEL_FUNCTIONS.get(model_name)
            if func is None:
                continue
            model_answer, elapsed = func(image_path, q["question"])
            score = judge_answer(q["question"], q["answer"], model_answer)
            row[f"{model_name} 回答"] = model_answer[:30] + "..."
            row[f"{model_name} スコア"] = score

        rows.append(row)

    if not rows:
        return "<p>データがありません。</p>", "<p></p>"

    th_style = (
        f"border:1px solid {COLOR_BORDER}; padding:8px; "
        f"background:{COLOR_HEADER_BG}; color:{COLOR_HEADER_FG}; "
        "text-align:left; font-weight:bold;"
    )
    td_style = (
        f"border:1px solid {COLOR_BORDER}; padding:6px; color:{COLOR_CELL_FG};"
    )
    badge_style_base = (
        "color:#ffffff; padding:3px 10px; border-radius:8px; font-weight:bold;"
    )

    # 結果テーブルHTML
    headers = list(rows[0].keys())
    table_html = "<table style='width:100%; border-collapse:collapse; font-size:0.85em;'>"
    table_html += "<tr>" + "".join(f"<th style='{th_style}'>{h}</th>" for h in headers) + "</tr>"
    for row in rows:
        table_html += "<tr>" + "".join(
            f"<td style='{td_style}'>{row.get(h, '')}</td>" for h in headers
        ) + "</tr>"
    table_html += "</table>"

    # サマリー
    summary_parts = []

    # モデル別平均スコア
    summary_parts.append(f"<h3 style='color:{COLOR_HEADER_FG};'>モデル別平均スコア</h3>")
    summary_parts.append("<table style='border-collapse:collapse; margin-bottom:16px;'>")
    summary_parts.append(
        f"<tr><th style='{th_style}'>モデル</th>"
        f"<th style='{th_style}'>平均スコア</th>"
        f"<th style='{th_style}'>評価数</th></tr>"
    )
    for model_name in selected_models:
        key = f"{model_name} スコア"
        scores = [r[key] for r in rows if key in r and r[key] > 0]
        avg = sum(scores) / len(scores) if scores else 0
        color = SCORE_COLORS.get(round(avg), COLOR_MUTED)
        summary_parts.append(
            f"<tr><td style='{td_style}'>{model_name}</td>"
            f"<td style='{td_style}'>"
            f"<span style='background:{color}; {badge_style_base}'>{avg:.2f}</span></td>"
            f"<td style='{td_style}'>{len(scores)}</td></tr>"
        )
    summary_parts.append("</table>")

    # 画像タイプ別 × モデル別
    summary_parts.append(f"<h3 style='color:{COLOR_HEADER_FG};'>画像タイプ別 × モデル別 平均スコア</h3>")
    types_in_results = sorted(set(r["画像タイプ"] for r in rows))
    summary_parts.append("<table style='border-collapse:collapse;'>")
    summary_parts.append(
        f"<tr><th style='{th_style}'>画像タイプ</th>"
        + "".join(f"<th style='{th_style}'>{m}</th>" for m in selected_models)
        + "</tr>"
    )
    for t in types_in_results:
        type_rows = [r for r in rows if r["画像タイプ"] == t]
        summary_parts.append(
            f"<tr><td style='{td_style} font-weight:bold;'>{t}</td>"
        )
        for model_name in selected_models:
            key = f"{model_name} スコア"
            scores = [r[key] for r in type_rows if key in r and r[key] > 0]
            avg = sum(scores) / len(scores) if scores else 0
            color = SCORE_COLORS.get(round(avg), COLOR_MUTED)
            summary_parts.append(
                f"<td style='{td_style} text-align:center;'>"
                f"<span style='background:{color}; {badge_style_base}'>{avg:.2f}</span></td>"
            )
        summary_parts.append("</tr>")
    summary_parts.append("</table>")

    return table_html, "".join(summary_parts)


# ---------- UI構築 ----------

AVAILABLE = get_available_models()
MODEL_CHOICES = [name for name, available in AVAILABLE.items() if available]
UNAVAILABLE_MODELS = [name for name, available in AVAILABLE.items() if not available]


def build_app():
    with gr.Blocks(
        title="BusinessSlideVQA Evaluator",
    ) as app:
        gr.Markdown(
            "# BusinessSlideVQA Evaluator - フロンティアVLM比較\n"
            "ストックマーク社の[BusinessSlideVQA](https://github.com/stockmarkteam/business-slide-questions)"
            "ベンチマークを使い、フロンティアVLMの日本語ビジネスドキュメント読解精度を比較評価します。\n\n"
            f"**対応モデル:** {', '.join(MODEL_CHOICES)}"
            + (f"  |  **APIキー未設定:** {', '.join(UNAVAILABLE_MODELS)}" if UNAVAILABLE_MODELS else "")
        )

        with gr.Tab("1問ずつ評価"):
            with gr.Row():
                type_filter = gr.Dropdown(
                    choices=["全て"] + IMAGE_TYPES,
                    value="全て",
                    label="画像タイプ",
                    scale=1,
                )
                question_select = gr.Dropdown(
                    choices=[f"Q{q['question_id']}: {q['question'][:20]}..." for q in DATASET],
                    label="問題選択",
                    scale=3,
                )
            with gr.Row():
                model_select = gr.CheckboxGroup(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES,
                    label="使用モデル",
                )
                run_btn = gr.Button("実行", variant="primary", scale=0)

            with gr.Row():
                image_display = gr.Image(label="スライド画像", scale=2)
                with gr.Column(scale=1):
                    question_display = gr.Textbox(label="質問", interactive=False)
                    answer_display = gr.Textbox(label="正解", interactive=False)

            results_display = gr.HTML(label="モデル別回答")

            type_filter.change(
                fn=get_filtered_questions,
                inputs=[type_filter],
                outputs=[question_select],
            )

            run_btn.click(
                fn=run_single_eval,
                inputs=[type_filter, question_select, model_select],
                outputs=[image_display, question_display, answer_display, results_display],
            )

        with gr.Tab("バッチ評価 & サマリー"):
            with gr.Row():
                batch_type = gr.Dropdown(
                    choices=["全て"] + IMAGE_TYPES,
                    value="全て",
                    label="画像タイプ",
                    scale=1,
                )
                batch_models = gr.CheckboxGroup(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES,
                    label="使用モデル",
                    scale=2,
                )
                batch_btn = gr.Button("バッチ実行", variant="primary", scale=0)

            batch_results = gr.HTML(label="結果テーブル")
            batch_summary = gr.HTML(label="サマリー")

            batch_btn.click(
                fn=run_batch_eval,
                inputs=[batch_type, batch_models],
                outputs=[batch_results, batch_summary],
            )

        with gr.Tab("About"):
            gr.Markdown("""
## BusinessSlideVQA について

[BusinessSlideVQA](https://github.com/stockmarkteam/business-slide-questions) は、
ストックマーク株式会社が公開した **日本語ビジネスドキュメント向けVLMベンチマーク** です。

- 公開資料のスライド画像 220 問の質問応答ペアで構成
- テキスト、表、棒グラフ、折れ線グラフ、円グラフ、積み上げ棒グラフ、画像、その他の8カテゴリ
- 日本語のビジネス文脈における視覚的言語理解能力を評価

## 使用モデル

| モデル | プロバイダ | API |
|--------|-----------|-----|
| Claude Sonnet 4.6 | Anthropic | Anthropic API |
| GPT-4.1 | OpenAI | OpenAI API |
| Qwen2.5-VL-72B | Alibaba (via OpenRouter) | OpenRouter API |

## 評価方法: LLM-as-a-Judge

各モデルの回答を **Claude Sonnet 4.6** が審判として 1〜5 のスコアで評価します。

| スコア | 意味 |
|--------|------|
| 5 | 完全に正解 |
| 4 | ほぼ正解 |
| 3 | おおむね正しいが不完全 |
| 2 | 一部正しいが重要な誤りあり |
| 1 | 完全に不正解 |

## このアプリの目的

フロンティアVLMの日本語ビジネスドキュメント読解能力を **手軽に比較評価** し、
各モデルの得意・不得意を可視化することを目的としています。

## クレジット

- **BusinessSlideVQA**: Stockmark Inc. ([GitHub](https://github.com/stockmarkteam/business-slide-questions))
- **ライセンス**: MIT
            """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)
