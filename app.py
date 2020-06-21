# -*- coding: utf-8 -*-
"""

@author: Adith George
"""

import numpy as np
import pandas as pd
import pickle
import pandas as pd

#from flasgger import Swagger

import streamlit as st 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

class preprocess_text(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        
        if isinstance(X,pd.Series):
            X = X.copy()
            X = X.str.replace(r"http\S+", "")
            X = X.str.replace(r"http", "")
            X = X.str.replace(r"@\S+", "")
            X = X.str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
            X = X.str.replace(r"@", "at")
            X = X.str.lower()
            return X
        

pickle_in = open("pipe.pkl","rb")
pipeline = pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_tweet_sentiment(text):
    
    """ Let's Predict whether Apple tweet sentiment is positive or negative 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: text
        in: query
        type: string
        required: true
    
    """
   
    prediction = pipeline.predict(pd.Series(text))
   
    print(prediction)
    
    return prediction



def main():
    
    #st.title("Apple Tweet Sentiment Prediction")
    
    html_temp = """
    <div style="background-color:#4682B4;padding:10px">
    <h2 style="color:white;text-align:center;"> Apple Tweet Sentiment Prediction </h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
   
    text = st.text_area("Tweet","Type Here")
    
    result=""
    tweet = ""
    
    if st.button("Predict"):
        result = predict_tweet_sentiment(text)
    
    if result == 1:
        tweet = 'positive'
    elif result == 0:
        tweet = 'negative'
    
    st.success('The tweet is {}'.format(tweet))
    
    
    if st.button("About"):
        st.text("This is a tweet sentiment predictor for Apple based tweets.")
        st.text("You may find the predictor a little biased.")
        st.text("The predictor will be further trained on a variety of data to improve accuracy.")

if __name__=='__main__':
    main()
    
    
    