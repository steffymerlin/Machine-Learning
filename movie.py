import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
nltk.download('stopwords')
nltk.download('punkt')
train_path = r"C:\Users\Admin\Desktop\movie\train_data.txt"
train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords_set = set(stopwords.words('english'))
    text = " ".join([i for i in words if i not in stopwords_set and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()
    return text
train_data['Text_cleaning'] = train_data['Description'].apply(clean_text)
tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(train_data['Text_cleaning'])
y = train_data['Genre']
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))
def predict_genre(description):
    cleaned_description = clean_text(description)
    description_tfidf = tfidf_vectorizer.transform([cleaned_description])
    predicted_genre = classifier.predict(description_tfidf)
    return predicted_genre[0]
new_description = "A young boy discovers he has magical powers and attends a school for wizards."
predicted_genre = predict_genre(new_description)
print(f'The predicted genre for the new description is: {predicted_genre}')
