import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import data_processing
import pandas as pd
import sklearn

class Playstore_app_review_sentiment_analyser():
    def __init__(self):
        self.dataset = data_processing.return_processed_df()
        c_value = 200
        #print(f"c_value = {c_value}")
        self.model = sklearn.linear_model.LogisticRegression(C=c_value)
        self.vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,3))
    
    def train(self):
        train_df , test_df = sklearn.model_selection.train_test_split(self.dataset, train_size=0.8, test_size=0.2, random_state=42)

        X_train = self.vectoriser.fit_transform(train_df["Translated_Review"])
        y_train = train_df["Sentiment"]

        self.model.fit(X_train, y_train)

        X_test = self.vectoriser.transform(test_df["Translated_Review"])
        y_true = test_df["Sentiment"]
        y_predicted = self.model.predict(X_test)
        print(self.vectoriser.get_feature_names_out())
        return sklearn.metrics.classification_report(y_true, y_predicted, output_dict=True)

    def predict(self, review:str):
        X_predict = self.vectoriser.transform([review])
        y_pred = self.model.predict_proba(X_predict)
        return list(y_pred)

m = Playstore_app_review_sentiment_analyser()
m.train()
        

""" metrics = ['precision', 'recall', 'f1-score']
labels = ['-1', '0', '1', 'macro avg', 'weighted avg']

accumulator = {label: {metric: 0 for metric in metrics} for label in labels}

for i in range(10):
    model = Playstore_app_review_sentiment_analyser()
    report = model.train()

    for label in labels:
        for metric in metrics:
            accumulator[label][metric] += report[label][metric]

# Compute averages
print("\nAverage over 10 runs:\n")
for label in labels:
    print(f"Label {label}:")
    for metric in metrics:
        avg_value = accumulator[label][metric] / 10
        print(f"  {metric:>9}: {avg_value:.4f}") """

