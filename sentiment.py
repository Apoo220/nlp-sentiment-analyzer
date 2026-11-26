"""Sentiment analysis for Japanese reviews.

Scores text with a sentiment dictionary after Janome tokenizing.
Negation flips a nearby sentiment word, intensifiers boost one.
"""

from janome.tokenizer import Tokenizer

_tokenizer = Tokenizer()

# --- Sentiment dictionary -------------------------------------------------
# weight: +2 strong positive, +1 positive, -1 negative, -2 strong negative
POSITIVE = {
    "美味しい": 2, "おいしい": 2, "最高": 2, "素晴らしい": 2, "大好き": 2,
    "感動": 2, "絶品": 2, "完璧": 2, "幸せ": 1, "嬉しい": 1, "楽しい": 1,
    "良い": 1, "いい": 1, "よい": 1, "好き": 1, "満足": 1, "快適": 1,
    "丁寧": 1, "親切": 1, "きれい": 1, "綺麗": 1, "清潔": 1, "便利": 1,
    "安い": 1, "早い": 1, "速い": 1, "新しい": 1, "広い": 1, "明るい": 1,
    "優しい": 1, "上手": 1, "おすすめ": 1, "オススメ": 1, "コスパ": 1,
    "ふわふわ": 1, "もちもち": 1, "ジューシー": 1, "新鮮": 1, "丁度いい": 1,
    "ちょうどいい": 1, "気持ちいい": 1, "落ち着く": 1, "静か": 1,
    "また来たい": 2, "リピート": 1, "笑顔": 1, "元気": 1, "安心": 1,
    "助かる": 1, "ありがたい": 1, "丁寧な": 1, "感じがいい": 1,
}

NEGATIVE = {
    "まずい": -2, "最悪": -2, "ひどい": -2, "酷い": -2, "がっかり": -2,
    "残念": -1, "悪い": -1, "嫌い": -1, "不満": -1, "遅い": -1,
    "高い": -1, "うるさい": -1, "汚い": -1, "狭い": -1, "暗い": -1,
    "冷たい": -1, "冷める": -1, "固い": -1, "硬い": -1, "油っぽい": -1,
    "しょっぱい": -1, "辛すぎる": -1, "薄い": -1, "臭い": -1, "臭う": -1,
    "待たされる": -1, "混雑": -1, "混んでいる": -1, "行列": -1,
    "不親切": -1, "無愛想": -1, "雑": -1, "面倒": -1, "不便": -1,
    "痛い": -1, "疲れる": -1, "退屈": -1, "つまらない": -1, "微妙": -1,
    "ぬるい": -1, "ベタベタ": -1, "パサパサ": -1, "二度と": -2,
    "失敗": -1, "不衛生": -2, "虫": -1, "髪の毛": -1,
}

SENTIMENT = {**POSITIVE, **NEGATIVE}

NEGATIONS = {"ない", "無い", "ぬ", "ん", "じゃない", "ではない", "なく"}
INTENSIFIERS = {"とても", "すごく", "凄く", "非常に", "かなり", "本当に",
                "めちゃくちゃ", "超", "一番", "すごい"}
DIMINISHERS = {"少し", "ちょっと", "やや", "あまり"}


def _tokens(text):
    return list(_tokenizer.tokenize(text))


def _surface(tok):
    return tok.surface


def _base(tok):
    return tok.base_form if tok.base_form != "*" else tok.surface


def analyze(text):
    """Return dict: label, score, positive hits, negative hits, key phrases."""
    toks = _tokens(text)
    surfaces = [_surface(t) for t in toks]
    bases = [_base(t) for t in toks]

    score = 0.0
    hits = []  # (surface, weight) actually applied
    multiplier = 1.0
    pending_flip = False

    for i, tok in enumerate(toks):
        word = bases[i]
        surf = surfaces[i]

        if surf in INTENSIFIERS or word in INTENSIFIERS:
            multiplier = 1.5
            continue
        if surf in DIMINISHERS or word in DIMINISHERS:
            multiplier = 0.5
            continue

        weight = SENTIMENT.get(word, SENTIMENT.get(surf, 0))
        if weight:
            w = weight * multiplier
            if pending_flip:
                w = -w
                pending_flip = False
            score += w
            hits.append((surf, w))
            multiplier = 1.0
            continue

        if surf in NEGATIONS or word in NEGATIONS:
            # Flip the most recent sentiment hit if one is close behind,
            # otherwise mark that the next sentiment word is negated.
            if hits and i - _last_hit_index(toks, hits) <= 3:
                s, w = hits[-1]
                score -= 2 * w
                hits[-1] = (s, -w)
            else:
                pending_flip = True

    if score >= 1.0:
        label = "ポジティブ"
    elif score <= -1.0:
        label = "ネガティブ"
    else:
        label = "中立"

    return {
        "text": text,
        "label": label,
        "score": round(score, 2),
        "hits": hits,
        "key_phrases": extract_key_phrases(toks, hits),
    }


def _last_hit_index(toks, hits):
    # Find where the most recent sentiment word was, for the negation check.
    target = hits[-1][0]
    for i in range(len(toks) - 1, -1, -1):
        if toks[i].surface == target:
            return i
    return 0


def extract_key_phrases(toks, hits, top_n=5):
    """Key phrases: noun sequences (2+ nouns) by frequency, plus the
    sentiment-bearing words themselves."""
    from collections import Counter

    hit_words = [s for s, _ in hits]

    noun_runs = []
    run = []
    for tok in toks:
        pos = tok.part_of_speech.split(",")[0]
        if pos == "名詞":
            run.append(tok.surface)
        else:
            if len(run) >= 2:
                noun_runs.append("".join(run))
            run = []
    if len(run) >= 2:
        noun_runs.append("".join(run))

    freq = Counter(noun_runs)
    phrases = [p for p, _ in freq.most_common(top_n)]
    merged = hit_words + [p for p in phrases if p not in hit_words]
    return merged[:top_n]
