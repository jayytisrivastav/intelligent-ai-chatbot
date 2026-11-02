# Intelligent AI Chatbot

A Python command-line assistant for reserving a lab microscope. A user can ask
“Can I book the microscope tomorrow?”, give the time and duration in a follow-up
message, and confirm the reservation to receive a booking ID.

## How it works

- A TensorFlow/Keras neural network classifies messages into four intents:
  greeting, thanks, microscope booking, and cancellation.
- The booking flow remembers the date, start time, and duration across messages.
- SQLite stores confirmed reservations between sessions and prevents overlapping
  bookings. A user can look up a reservation with `show M1` or request
  cancellation with `cancel M1`. Cancellation requires confirmation.

## Run locally

From the project folder, run:

```powershell
python -m pip install tensorflow numpy nltk scikit-learn
python -m nltk.downloader wordnet
python training.py
python chatbot.py
```

`training.py` reads `intents.json` and saves the model, vocabulary, and intent
labels. If those files are already present, you can launch `chatbot.py` directly.
Type `quit` to end the session.

## Evaluation

Run `python evaluate.py` to compare the neural classifier with a TF-IDF +
logistic regression baseline. Both use the same training patterns.

| Model | Correct on 12 test requests | Macro F1 |
|---|---:|---:|
| Neural classifier | 11/12 (91.7%) | 0.914 |
| TF-IDF + logistic regression | 8/12 (66.7%) | 0.631 |

These results come from a small, hand-written test set with three examples per
intent. The neural model misclassified “I need a microscope slot” as cancellation.

A separate test checked whether the classifier rejected unrelated requests.
It rejected 1 of 4. It accepted a train-ticket request, a pizza cancellation,
and a weather question as microscope-related intents. These tests measure
intent predictions; they do not create or cancel reservations.

## Project background

I built the microscope reservation workflow, including multi-turn booking,
SQLite storage, overlap detection, and cancellation by ID. I also evaluated
the intent classifier against a TF-IDF and logistic regression baseline. 
The initial intent-classification approach was adapted from NeuralNine’s 
*Intelligent AI Chatbot in Python*.
