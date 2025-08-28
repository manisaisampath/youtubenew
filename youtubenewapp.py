import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle as pk
nltk.download("stopwords")
model=pk.load(open("model.pkl","rb"))
vector=pk.load(open("vector.pkl","rb"))
st.set_page_config(page_title="YouTube Sentiment Analysis", page_icon="🎬", layout="centered")
st.title("YouTube Video Comments Sentiment Analysis")
st.write("This application predicts the sentiment of YouTube video comments as positive or negative.")
port=PorterStemmer()
def steming(context):
  stemmed=re.sub("[^a-zA-z]"," ",context)
  stemmed=stemmed.lower()
  stemmed=stemmed.split()
  stemmed=[port.stem(word) for word in stemmed if word not in stopwords.words("english")]
  stemmed=" ".join(stemmed)
  return stemmed
def predicting(text):
    text=steming(text)
    text=vector.transform([text])
    result=model.predict(text)
    return result[0]
comment=st.text_input("Enter the YouTube video comment:")

st.button("Predict")
if comment.strip()=="":
    st.error("Please enter a comment to predict sentiment.")

else:
    result=predicting(comment)
    if result==1:
        st.success("positive")
    else:
        st.error("negative")

