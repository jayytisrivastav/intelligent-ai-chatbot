import json
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

# Find intents.json in the same folder as this Python file.
project_folder = Path(__file__).resolve().parent

with open(project_folder / "intents.json", encoding="utf-8") as file:
    intents = json.load(file)


words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]


for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Split each example sentence into tokens.
        word_list = nltk.word_tokenize(pattern, preserve_line=True)

        # Add individual tokens to our vocabulary list.
        words.extend(word_list)

        # Store the sentence tokens together with their intent label.
        documents.append((word_list, intent["tag"]))

        # Keep each intent label only once.
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


print("Intent labels:")
print(classes)

print("\nTraining examples:")
for document in documents:
    print(document)

print("\nNumber of examples:", len(documents))

import pickle
import random

import numpy as np
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


# Clean the vocabulary, remove duplicates, and fix its order.
words = sorted(set(
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_letters
))

classes = sorted(set(classes))


# Convert each example into input numbers and a target label.
training = []

for token_list, tag in documents:
    cleaned_words = {
        lemmatizer.lemmatize(word.lower())
        for word in token_list
        if word not in ignore_letters
    }

    # One number per vocabulary word: present = 1, absent = 0.
    bag = [
        1 if word in cleaned_words else 0
        for word in words
    ]

    # One number per intent: correct intent = 1, others = 0.
    output_row = [0] * len(classes)
    output_row[classes.index(tag)] = 1

    training.append((bag, output_row))


random.shuffle(training)

# Build separate rectangular arrays for inputs and labels.
train_x = np.array(
    [example[0] for example in training],
    dtype=np.float32
)

train_y = np.array(
    [example[1] for example in training],
    dtype=np.float32
)


# Build the neural network.
model = Sequential([
    Input(shape=(len(words),)),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Train the model.
model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=5,
    verbose=1
)


# Save the vocabulary and labels in their exact training order.
with open(project_folder / "words.pkl", "wb") as file:
    pickle.dump(words, file)

with open(project_folder / "classes.pkl", "wb") as file:
    pickle.dump(classes, file)

model.save(project_folder / "chatbot_model.keras")

print("\nTraining complete. Model, vocabulary, and labels saved.")