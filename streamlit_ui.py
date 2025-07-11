import streamlit as st
from playstore_app_review_sentiment_analyser import Playstore_app_review_sentiment_analyser
import pandas as pd

@st.cache_resource
def load_model():
    model = Playstore_app_review_sentiment_analyser()
    model.train()
    return model

# Loading the cached model
if "model" not in st.session_state:
    st.session_state.model = load_model()

st.write("#### I've made use of logistic regression to predict the sentiment of reviews of playstore applications")
st.write("\n\n\n\n")

review = st.text_input(label = "Enter a review of any application", key="user_input")
prob_list = st.session_state.model.predict(review)[0]
predictions = ["Negative", "Neutral", "Positive"]

if(prob_list[0] == max(prob_list)):
    st.write("Output : Negative 😔")
elif (prob_list[0] == max(prob_list)):
    st.write("Output : Neutral (•_•)")
else:
    st.write("Output : Positive 🤩")
if(st.checkbox(label="See the probability values")):
    df = pd.DataFrame([ [predictions[i], prob_list[i]] for i in range(3)], columns=["Prediction", "Probability"])
    st.dataframe(df)
