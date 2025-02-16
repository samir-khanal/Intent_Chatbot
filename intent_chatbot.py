import json
import random
import nltk
from nltk.stem import WordNetLemmatizer # Lemmatizer to reduce words to their base form
import re
from nltk.tokenize import word_tokenize  # Tokenizer to split text into words
from nltk.corpus import stopwords  # List of common stopwords to remove
import pandas as pd

# Download necessary NLTK data
nltk.download('punkt')# Tokenizer models
nltk.download('wordnet')
nltk.download('stopwords') # Stopword list

# ----- Load the Dataset -----
with open("intents.json", "r") as file:
    data = json.load(file)
# Initialize a lemmatizer to reduce words to their base form.
lemmatizer = WordNetLemmatizer()

# tokenizer and lemmatizer function
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]
#This function tokenizes a text string into words and converts each word to lowercase and lemmatizes it.


# Preparing the training data
# Creating empty lists to hold training(patterns) and their intent tags.
patterns = []
tags = []
# Loop through each intent in the dataset
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)   # Append the pattern to the training sentences
        tags.append(intent["tag"])  # Append the corresponding tag

# Convert to DataFrame
df = pd.DataFrame({'text': patterns, 'label': tags})
df.head(15)

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_lemmatize)
# Transforming the list of patterns into numerical features
X = vectorizer.fit_transform(df['text'])
y = df['label']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# function to get chatbot response
def predict_intent(user_input):
    # Converting to TF-IDF features
    text_vec = vectorizer.transform([user_input])
    # Predicting the intent
    intent = model.predict(text_vec)[0]
    # finding a random response for the predicted intent
    for item in data["intents"]:
        if item["tag"] == intent:
            return random.choice(item["responses"])
    return "I'm sorry, I didn't understand that."

# Test the model with a sample input
sample_input = "What hour?"
print("Predicted Response:", predict_intent(sample_input))