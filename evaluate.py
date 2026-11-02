import json
import runpy
from pathlib import Path
from unittest.mock import patch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

folder = Path(__file__).resolve().parent

# Load the actual chatbot functions. Answer "quit" once to exit its chat loop.
with patch("builtins.input", return_value="quit"):
    chatbot = runpy.run_path(str(folder / "chatbot.py"))

predict_class = chatbot["predict_class"]

with open(folder / "intents.json", encoding="utf-8") as file:
    intents = json.load(file)["intents"]

# Train the baseline on the same patterns used by the neural network.
patterns = []
labels = []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])

baseline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000, random_state=42),
)
baseline.fit(patterns, labels)

# Keep these examples out of intents.json so they remain test cases.
test_cases = [
    ("Hello there", "greetings"),
    ("Hey, good morning", "greetings"),
    ("Hi, good evening", "greetings"),
    ("Many thanks", "thanks"),
    ("Thanks for helping me", "thanks"),
    ("Thank you for the help", "thanks"),
    ("Can I reserve the microscope tomorrow?", "book_microscope"),
    ("I need a microscope slot", "book_microscope"),
    ("Please book a slot for me", "book_microscope"),
    ("Please cancel my reservation", "cancel_booking"),
    ("Cancel the microscope slot I booked", "cancel_booking"),
    ("I no longer need my booking", "cancel_booking"),
]

expected = [label for _, label in test_cases]
sentences = [sentence for sentence, _ in test_cases]
baseline_labels = baseline.predict(sentences)
neural_labels = []

for sentence, correct in test_cases:
    predictions = predict_class(sentence)
    actual = predictions[0]["intent"] if predictions else "fallback"
    neural_labels.append(actual)
    if actual != correct:
        print(f"Neural model missed {sentence!r}: expected {correct}, got {actual}")

intent_labels = sorted(set(expected))
for name, results in [("Neural chatbot", neural_labels), ("TF-IDF baseline", baseline_labels)]:
    score = f1_score(expected, results, labels=intent_labels,
                     average="macro", zero_division=0)
    hits = sum(actual == correct for actual, correct in zip(results, expected))
    print(f"{name}: {hits}/{len(expected)} correct | Macro F1: {score:.3f}")

# Separately check whether the chatbot rejects unsupported requests.
unrelated = [
    "How do I bake a cake?",
    "Can I book a train ticket?",
    "Cancel my pizza order",
    "What is the weather tomorrow?",
]
rejected = 0
for sentence in unrelated:
    predictions = predict_class(sentence)
    if predictions:
        print(f"False acceptance: {sentence!r} -> {predictions[0]['intent']}")
    else:
        rejected += 1

print(f"Unrelated requests rejected: {rejected}/{len(unrelated)}")