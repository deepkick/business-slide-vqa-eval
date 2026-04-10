---
title: BusinessSlideVQA Evaluator
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
license: mit
---

# BusinessSlideVQA Evaluator

フロンティアVLM（Claude Sonnet 4.6 / GPT-4.1 / Qwen2.5-VL-72B）の日本語ビジネスドキュメント読解精度を比較評価するGradio Webアプリ。

[BusinessSlideVQA](https://github.com/stockmarkteam/business-slide-questions)（ストックマーク社）のベンチマークを使用。

## 評価結果

画像タイプ別に均等サンプリングした30問に対するバッチ評価結果（評価方法: Claude Sonnet 4.6によるLLM-as-a-Judge、1〜5スコア）。

### モデル別平均スコア（30問）

| モデル | 平均スコア |
|--------|---------|
| Claude Sonnet 4.6 | **4.73** |
| GPT-4.1 | 4.43 |
| Qwen2.5-VL-72B | 4.43 |

### 画像タイプ別 × モデル別 平均スコア

| 画像タイプ | Claude Sonnet 4.6 | GPT-4.1 | Qwen2.5-VL-72B |
|-----------|--------|---------|------------|
| テキスト | 5.00 | 4.50 | 4.25 |
| 円グラフ | 5.00 | 5.00 | 5.00 |
| 棒グラフ | 5.00 | 3.75 | 5.00 |
| 積み上げ棒グラフ | 5.00 | 4.67 | 3.67 |
| 表 | 4.80 | 4.80 | 3.80 |
| 画像 | 4.60 | 4.60 | 4.60 |
| 折れ線グラフ | 4.00 | 3.33 | 4.67 |
| その他 | 4.33 | 4.67 | 4.67 |

### 主な知見

- **Claude Sonnet 4.6** が総合トップ。特にテキスト・棒グラフ・積み上げ棒グラフで満点
- **GPT-4.1** は棒グラフの数値読み取りで精度が落ちる傾向
- **Qwen2.5-VL-72B** は表の構造理解がやや弱いが、折れ線グラフではClaudeを上回る
- **円グラフ** は全モデル満点で、比較的容易なタスク
- 評価方法: Claude Sonnet 4.6によるLLM-as-a-Judge（1-5スコア）

## セットアップ

```bash
# 1. リポジトリクローン
git clone <this-repo>
cd business-slide-vqa-eval

# 2. 仮想環境
python -m venv .venv
source .venv/bin/activate

# 3. 依存パッケージ
pip install -r requirements.txt

# 4. APIキー設定
cp .env.example .env
# .env を編集してAPIキーを設定

# 5. データ準備（BusinessSlideVQAリポジトリが ../business-slide-questions/ にある前提）
python prepare_data.py

# 6. 起動
python app.py
```

## HuggingFace Spaces デプロイ

1. HF Spaces で新しい Gradio Space を作成
2. コードをプッシュ
3. Settings > Secrets に `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY` を設定
4. `data/images/` は HF Dataset としてアップロードするか、初回起動時にダウンロードする仕組みに変更

## 評価方法

Claude Sonnet 4.6 を LLM-as-a-Judge として使用し、各モデルの回答を 1〜5 のスコアで評価。

## クレジット

- **BusinessSlideVQA**: Stockmark Inc.
- **ライセンス**: MIT
