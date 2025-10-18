import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.pipeline import Pipeline

# Load your dataset
df = pd.read_csv("Resume.csv")
X = df['Resume_str']
y = df['Category']

# Split into train/val/test (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocess and vectorize
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Train model
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
clf.fit(X_train_vec, y_train_enc)

# Evaluate
val_pred = clf.predict(X_val_vec)
print("Validation Accuracy:", accuracy_score(y_val_enc, val_pred))

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', clf)
])

with open("trained_models/ensemble_resume_classifier.pkl", "wb") as f:
    pickle.dump({
        'model': pipeline,
        'label_encoder': le
    }, f)