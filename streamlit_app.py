import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import time  # For adding delays if needed

# Download NLTK data
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
@st.cache_data(show_spinner=False)
def scrape_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract text from paragraph tags
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return ""

# Scrape and combine text from all URLs
@st.cache_data(show_spinner=True)
def build_corpus(urls):
    corpus_text = ""
    for url in urls:
        text = scrape_text(url)
        corpus_text += text + " "
        time.sleep(1)  # Be polite and avoid overwhelming the server
    return corpus_text

corpus_text = build_corpus(urls)

# Check if corpus_text is empty
if not corpus_text.strip():
    st.error("Failed to build the corpus from the provided URLs.")
    st.stop()

# Tokenize the text into sentences
sent_tokens = nltk.sent_tokenize(corpus_text)

# Initialize lemmatizer and stopwords
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Functions for text normalization and lemmatization
def LemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        token for token in tokens if token not in string.punctuation and token not in stop_words
    ]
    return LemTokens(tokens)

# Define TF-IDF vectorizer
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

# Define greeting inputs and responses
GREETING_INPUTS = (
    "hello", "hi", "hey", "hola", "greetings", "what's up", "howdy",
    "yo", "hi there", "good day", "morning", "afternoon", "evening",
    "sup", "yo yo", "hey there", "nice to meet you", "hello there",
    "what's happening", "how's it going"
)
GREETING_RESPONSES = [
    "Hello!", "Hi!", "Hey!", "Hola!", "Greetings!", "What's up?",
    "Howdy!", "Yo!", "Hi there!", "Good day!", "Good morning!",
    "Good afternoon!", "Good evening!", "Sup!", "Yo yo!", "Hey there!",
    "Nice to meet you!", "Hello there!", "Not much, you?",
    "It's going well, thanks!"
]

# Greeting function
def greeting(sentence):
    for word in sentence.lower().split():
        if word in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

# Response generation function
def generate_response(user_input):
    user_input = user_input.lower()
    bot_response = ''
    sent_tokens.append(user_input)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    if req_tfidf == 0:
        bot_response = "I'm sorry, I didn't understand that. Could you please rephrase?"
    else:
        bot_response = sent_tokens[idx]
    sent_tokens.pop()  # Remove the user input from tokens
    return bot_response

# Streamlit UI setup
st.title("🗨️ Simple Chatbot")
st.write("This chatbot provides information based on scraped Wikipedia articles.")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.text_input("You:", key="user_input")

if st.button("Send", key="send_button"):
    if user_input:
        st.session_state.chat_history.append(("You", user_input))
        greet_response = greeting(user_input)
        if greet_response:
            bot_response = greet_response
        else:
            bot_response = generate_response(user_input)
        st.session_state.chat_history.append(("Bot", bot_response))
        st.experimental_rerun()

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
