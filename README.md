# BusinessSlideVQA Evaluator

フロンティアVLM（Claude Sonnet 4.6 / GPT-4.1 / Qwen2.5-VL-72B）の日本語ビジネスドキュメント読解精度を比較評価するGradio Webアプリ。

[BusinessSlideVQA](https://github.com/stockmarkteam/business-slide-questions)（ストックマーク社）のベンチマークを使用。

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
