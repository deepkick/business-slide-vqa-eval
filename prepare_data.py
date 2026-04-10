"""BusinessSlideVQAデータセットから評価用サブセットを準備するスクリプト."""

import json
import random
import shutil
from pathlib import Path

SEED = 42
SOURCE_DIR = Path(__file__).resolve().parent.parent / "business-slide-questions"
VQA_JSON = SOURCE_DIR / "vqa.json"
IMAGES_SRC = SOURCE_DIR / "downloads" / "pngs"

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
SUBSET_JSON = OUTPUT_DIR / "vqa_subset.json"
IMAGES_DST = OUTPUT_DIR / "images"

# 画像タイプ別のサンプリング数
SAMPLE_COUNTS = {
    "テキスト": 4,
    "表": 5,
    "棒グラフ": 4,
    "折れ線グラフ": 3,
    "円グラフ": 3,
    "積み上げ棒グラフ": 3,
    "画像": 5,
    "その他": 3,
}


def main():
    with open(VQA_JSON, encoding="utf-8") as f:
        all_questions = json.load(f)

    # 画像が実在する問題のみを対象にする
    available = [q for q in all_questions if (IMAGES_SRC / q["image"]).exists()]
    print(f"画像が存在する問題: {len(available)}/{len(all_questions)}")

    # 画像タイプ別にグルーピング
    by_type: dict[str, list] = {}
    for q in available:
        by_type.setdefault(q["image type"], []).append(q)

    rng = random.Random(SEED)
    subset = []
    for img_type, count in SAMPLE_COUNTS.items():
        candidates = by_type.get(img_type, [])
        sampled = rng.sample(candidates, min(count, len(candidates)))
        subset.extend(sampled)

    # question_idでソート
    subset.sort(key=lambda q: q["question_id"])

    print(f"サンプリング結果: {len(subset)}問")
    for img_type, count in SAMPLE_COUNTS.items():
        actual = sum(1 for q in subset if q["image type"] == img_type)
        print(f"  {img_type}: {actual}問")

    # 出力ディレクトリ準備
    IMAGES_DST.mkdir(parents=True, exist_ok=True)

    # 画像をコピー
    copied = 0
    for q in subset:
        src = IMAGES_SRC / q["image"]
        dst = IMAGES_DST / q["image"]
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  警告: 画像が見つかりません: {q['image']}")

    print(f"画像コピー: {copied}/{len(subset)}枚")

    # JSONを保存
    with open(SUBSET_JSON, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"保存完了: {SUBSET_JSON}")


if __name__ == "__main__":
    main()
