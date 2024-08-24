import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import streamlit as st

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# List of URLs to scrape
urls = [
    "https://en.wikipedia.org/wiki/Y._S._Jagan_Mohan_Reddy",
    "https://en.wikipedia.org/wiki/Narendra_Modi",
    # Add more URLs as needed
]

# Function to scrape text from a URL
def scrape_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ")
    else:
        st.error(f"Failed to scrape {url}")
        return ""

# Scrape and combine text from all URLs
corpus_text = ""
for url in urls:
    corpus_text += scrape_text(url) + " "

# Tokenize the text into sentences
sent_tokens = nltk.sent_tokenize(corpus_text)

# Initialize WordNet lemmatizer and stopwords
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Define functions for text normalization and lemmatization
def LemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
    return LemTokens(tokens)

# Define TF-IDF vectorizer
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

# Define greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "hey", "hola", "greetings", "what's up", "howdy", "yo", "hi there", "good day", "morning", "afternoon", "evening", "sup", "yo yo", "hey there", "nice to meet you", "hello there", "what's happening", "how's it going")
GREETING_RESPONSES = ["Hello!", "Hi!", "Hey!", "Hola!", "Greetings!", "What's up?", "Howdy!", "Yo!", "Hi there!", "Good day!", "Good morning!", "Good afternoon!", "Good evening!", "Sup!", "Yo yo!", "Hey there!", "Nice to meet you!", "Hello there!", "Not much, you?", "It's going well, thanks!"]

# Define the greeting function
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

# Define the response function
def response(user_response):
    bot_response = ''
    sent_tokens_copy = sent_tokens.copy()  # Preserve original sentences
    sent_tokens_copy.append(user_response)
    tfidf = TfidfVec.fit_transform(sent_tokens_copy)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        bot_response = "I'm sorry, I didn't understand that."
    else:
        bot_response = sent_tokens[idx]
    return bot_response

# Streamlit UI setup
st.title("Chatbot")
st.write("This is a simple chatbot application.")

# Chat history container
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input field for user input
user_input = st.text_input("You:", key="input")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(f"You: {user_input}")
        if greeting(user_input):
            bot_response = greeting(user_input)
        else:
            bot_response = response(user_input)
        st.session_state.chat_history.append(f"BOT: {bot_response}")

# Display chat history
for chat in st.session_state.chat_history:
    st.write(chat)
