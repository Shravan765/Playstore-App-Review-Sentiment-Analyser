import pandas as pd

def return_processed_df(csvfile: str = "gplay_review.csv"):
    import pandas as pd

    df = pd.read_csv(csvfile)
    reviews_ratings = df[["Translated_Review", "Sentiment"]].copy()

    # Drop rows with missing values
    reviews_ratings.dropna(subset=["Translated_Review", "Sentiment"], inplace=True)

    # Clean: lowercase + strip
    reviews_ratings["Sentiment"] = reviews_ratings["Sentiment"].str.lower().str.strip()
    reviews_ratings["Translated_Review"] = reviews_ratings["Translated_Review"].str.lower().str.strip()

    # Keep only valid sentiments
    sentiment_list = ["negative", "neutral", "positive"]
    reviews_ratings = reviews_ratings[reviews_ratings["Sentiment"].isin(sentiment_list)]

    # Mapping sentiment to numeric values
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    reviews_ratings["Sentiment"] = reviews_ratings["Sentiment"].map(sentiment_map)

    # Downsizing to remove bias
    sample_size = min(reviews_ratings["Sentiment"].value_counts())
    neg = reviews_ratings[ reviews_ratings["Sentiment"] == -1 ].sample(n = sample_size, random_state= 42)
    neu = reviews_ratings[ reviews_ratings["Sentiment"] == 0 ].sample(n = sample_size, random_state= 42)
    pos = reviews_ratings[ reviews_ratings["Sentiment"] == 1 ].sample(n = sample_size, random_state= 42)

    balanced_reviews_ratings = pd.concat([neg, neu, pos]).sample(random_state=42, frac= 1).reset_index(drop=True)
    return balanced_reviews_ratings


