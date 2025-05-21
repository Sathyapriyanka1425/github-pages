import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset (you can replace this with a real dataset like 'fake_or_real_news.csv')
data = {
    "text": [
        "The economy is booming and jobs are at an all-time high.",
        "Aliens landed in Paris last night and disappeared without a trace.",
        "New tax reforms will reduce burden on middle class families.",
        "Scientists discovered a secret portal to another dimension.",
        "Stock markets hit record highs after positive earnings reports."
    ],
    "label": [1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
}
df = pd.DataFrame(data)

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Step 4: Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 6: Predictions and Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
