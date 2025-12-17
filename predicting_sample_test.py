import joblib
import sklearn
print(sklearn.__version__)



model_path = r"C:\Users\haotr\fake-news-detection-dataset-English\final_model.joblib"
vectorizer_path = r"C:\Users\haotr\fake-news-detection-dataset-English\final_tfidf_vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

sample_texts = [
    "The economy is improving and jobs are being created.",
    "Breaking news: Celebrity involved in scandal!",
    "Scientists discover new species in the Amazon rainforest.",
    "Fake news site spreads misinformation about health."
]

sample_tfidf = vectorizer.transform(sample_texts)
predictions = model.predict(sample_tfidf)
probabilities = model.predict_proba(sample_tfidf)
for text, pred, prob in zip(sample_texts, predictions, probabilities):
    print(f"Text: {text}\nPredicted Label: {pred}, Probabilities: {prob}\n")


