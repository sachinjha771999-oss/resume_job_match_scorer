import re
import string
import nltk

# Download NLTK data (if not already present)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    """Clean text by removing special characters, extra spaces, etc."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text, n=20):
    """Extract top n keywords from text using TF-IDF (simplified)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=n, stop_words='english')
    try:
        X = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        return list(feature_names)
    except:
        return []

def tokenize_and_remove_stopwords(text):
    """Tokenize and remove stopwords"""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens