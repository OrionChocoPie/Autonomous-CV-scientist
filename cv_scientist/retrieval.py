"""Deterministic CV vocabulary for indexed English/Russian memory retrieval."""
import re


STOP_WORDS = set("a an and are as at be but by did do does for from how i in is it me no not of on or our the their there these this to was we were what when which why with would you already try false как что почему где когда это для при мы ли по на и в из с о об не the".split())
ALIASES = {
    "objects": "object", "объектов": "object", "объекты": "object", "объект": "object",
    "мелких": "small", "мелкие": "small", "маленьких": "small",
    "пропуски": "fn", "пропущенные": "fn", "negatives": "fn", "negative": "fn", "missed": "fn",
    "positives": "fp", "positive": "fp", "ложные": "fp", "срабатывания": "fp",
    "полнота": "recall", "полноты": "recall", "полноте": "recall",
    "точность": "precision", "точности": "precision",
    "регрессия": "regression", "регрессии": "regression", "ухудшение": "regression", "падение": "regression",
    "гипотеза": "hypothesis", "гипотезы": "hypothesis", "hypotheses": "hypothesis",
    "отклонена": "rejected", "отклонённая": "rejected", "подтверждена": "supported",
    "эксперимент": "experiment", "эксперименты": "experiment", "experiments": "experiment",
}


def terms(text):
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    return list(dict.fromkeys(ALIASES.get(token, token) for token in tokens
                             if len(token) > 1 and not token.isdecimal() and token not in STOP_WORDS))
