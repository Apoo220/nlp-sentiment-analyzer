"""Command-line entry point.

Usage:
  python main.py "レビュー文"          -> analyze one review
  python main.py --batch sample_reviews.csv
                                       -> score a labelled CSV and
                                          print accuracy
CSV format: text,label   (label: ポジティブ / ネガティブ / 中立)
"""

import csv
import sys

from sentiment import analyze


def show(result):
    print(f"文章: {result['text']}")
    print(f"判定: {result['label']}  (スコア: {result['score']:+.2f})")
    if result["hits"]:
        words = ", ".join(f"{s}({w:+.1f})" for s, w in result["hits"])
        print(f"感情ワード: {words}")
    if result["key_phrases"]:
        print(f"キーフレーズ: {', '.join(result['key_phrases'])}")


def batch(path):
    correct = 0
    total = 0
    misses = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result = analyze(row["text"])
            total += 1
            if result["label"] == row["label"]:
                correct += 1
            else:
                misses.append((row, result))
    print(f"{total}件中 {correct}件 正解 — 正解率 {correct / total:.0%}")
    for row, result in misses:
        print(f"  ✗ 期待: {row['label']} / 判定: {result['label']} "
              f"(スコア {result['score']:+.2f}) — {row['text']}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--batch":
        batch(sys.argv[2])
    elif len(sys.argv) == 2:
        show(analyze(sys.argv[1]))
    else:
        print(__doc__)
