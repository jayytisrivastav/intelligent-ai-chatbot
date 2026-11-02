import json
import pickle
import random
from pathlib import Path
from booking_flow import BookingFlow
from booking_store import initialize_store, lookup_booking, cancel_booking
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


project_folder = Path(__file__).resolve().parent
lemmatizer = WordNetLemmatizer()
ignore_letters = ["?", "!", ".", ","]

with open(project_folder / "intents.json", encoding="utf-8") as file:
    intents = json.load(file)

with open(project_folder / "words.pkl", "rb") as file:
    words = pickle.load(file)

with open(project_folder / "classes.pkl", "rb") as file:
    classes = pickle.load(file)

model = load_model(
    project_folder / "chatbot_model.keras",
    compile=False
)


def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence, preserve_line=True)

    return [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
        if word not in ignore_letters
    ]


def bag_of_words(sentence):
    sentence_words = set(clean_up_sentence(sentence))

    bag = [
        1 if word in sentence_words else 0
        for word in words
    ]

    return np.array(bag, dtype=np.float32)


def predict_class(sentence):
    tokens = set(clean_up_sentence(sentence))
    known_tokens = tokens.intersection(words)

    # Reject messages with too many unfamiliar words.
    if not tokens or len(known_tokens) / len(tokens) < 0.5:
        return []

    bow = bag_of_words(sentence)

    if not bow.any():
        return []

    probabilities = model.predict(
        np.array([bow]), verbose=0
    )[0]

    confidence_threshold = 0.70

    results = [
        {
            "intent": classes[index],
            "probability": float(score)
        }
        for index, score in enumerate(probabilities)
        if score >= confidence_threshold
    ]

    results.sort(
        key=lambda result: result["probability"],
        reverse=True
    )

    return results


def get_response(predictions):
    if not predictions:
        return "I'm not sure I understood. Could you rephrase that?"

    predicted_tag = predictions[0]["intent"]

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "I don't have a response for that intent yet."

initialize_store()
booking = BookingFlow()
waiting_for_cancel_id = False
pending_cancel_id = None

print("Chatbot ready! Type 'quit' to stop.")

while True:
    message = input("\nYou: ").strip()
    answer = message.lower()

    if answer in {"quit", "exit"}:
        print("Bot: Goodbye!")
        break

    if not message:
        continue

    if answer.startswith("show "):
        booking_id = message.split(maxsplit=1)[1]
        print("Bot:", lookup_booking(booking_id))
        continue

    # Once a booking has been shown, wait for the cancellation decision.
    if pending_cancel_id is not None:
        if answer in {"yes", "y", "confirm"}:
            print("Bot:", cancel_booking(pending_cancel_id))
            pending_cancel_id = None
        elif answer in {"no", "n", "never mind", "cancel"}:
            print("Bot: Okay, your booking is still reserved.")
            pending_cancel_id = None
        else:
            print("Bot: Type yes to cancel this booking or no to keep it.")
        continue

    # The existing booking flow handles its own follow-up messages.
    if booking.active:
        print("Bot:", booking.reply(message))
        continue

    # Accept either "cancel M2" or an ID after the bot asks for one.
    if answer.startswith("cancel ") or waiting_for_cancel_id:
        if answer in {"never mind", "nevermind", "cancel", "no"}:
            waiting_for_cancel_id = False
            print("Bot: Okay, I stopped the cancellation request.")
            continue

        candidate = message
        if answer.startswith("cancel "):
            candidate = message.split(maxsplit=1)[1]

        code = candidate.strip().upper()

        # A phrase such as "cancel my booking" should ask for the ID.
        if not (code.startswith("M") and code[1:].isascii() and code[1:].isdigit()):
            waiting_for_cancel_id = True
            print("Bot: Which booking? Type its ID, like M2, or type never mind.")
            continue

        details = lookup_booking(code)
        print("Bot:", details)

        if details.startswith("Booking "):
            pending_cancel_id = code
            waiting_for_cancel_id = False
            print("Bot: Cancel this booking? Type yes or no.")
        else:
            waiting_for_cancel_id = True
            print("Bot: Try another ID or type never mind.")
        continue

    predictions = predict_class(message)
    tag = predictions[0]["intent"] if predictions else None

    if tag == "book_microscope":
        response = booking.begin(message)
    elif tag == "cancel_booking" or answer == "cancel":
        waiting_for_cancel_id = True
        response = "Which booking would you like to cancel? Type its ID, like M2."
    else:
        response = get_response(predictions)

    print("Bot:", response)

    if predictions:
        best = predictions[0]
        print(
            f"Predicted intent: {best['intent']} "
            f"| Model score: {best['probability']:.2f}"
        )