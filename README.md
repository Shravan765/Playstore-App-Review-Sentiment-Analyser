# Playstore-App-Review-Sentiment-Analyser
A lightweight sentiment analysis tool that predicts whether a Play Store app review is Negative, Neutral, or Positive, built using Logistic Regression and TF-IDF vectorisation.  

Average over 10 runs (80:20 train - test data split):
Label -1 (Negative):  
  precision: 0.8083  recall: 0.8918  f1-score: 0.8480  
Label 0 (Neutral):  
  precision: 0.9241  recall: 0.8288  f1-score: 0.8738  
Label 1 (Positive):  
  precision: 0.8700  recall: 0.8725  f1-score: 0.8712  
Label macro avg:  
  precision: 0.8674  recall: 0.8644  f1-score: 0.8644  
Label weighted avg:  
  precision: 0.8680  recall: 0.8640  f1-score: 0.8644  
  
https://playstore-app-review-sentiment-analyser.streamlit.app/
