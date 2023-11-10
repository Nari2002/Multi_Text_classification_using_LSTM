import pickle
import sklearn
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model architecture
with open("model_architecture.json", 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
# Load model weights
loaded_model.load_weights("multi_text_model.h5")

# Load tokenizer
with open("tokenizer_text.pkl", 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

categories = ['Business', 'Entertainment', 'politics', 'Sports', 'Technology']    
    
def analysis(input_text):
    seq = loaded_tokenizer.texts_to_sequences(input_text)
    padded = pad_sequences(seq, maxlen=2000)
    prediction = loaded_model.predict(padded)
    # Get the prediction probabilities
    probabilities = prediction[0]
     
    
     # Define category labels
    category_labels =['Business', 'Entertainment', 'politics', 'Sports', 'Technology'] 
    
     # Create a DataFrame with the labels as the index
    data = pd.DataFrame({'Probability': probabilities}, index=category_labels)
    
     # Plot the probabilities using a bar chart with labels
    st.markdown('<h1 class="title">Probabilities for Text</h1>', unsafe_allow_html=True)
    st.bar_chart(data, use_container_width=True, width=600, height=400)
    predictions = np.argmax(prediction)
    
    if (predictions == 0):
        return prediction,st.markdown('<h1 class="title">Text is a bout Business</h1>', unsafe_allow_html=True),st.image("https://images.squarespace-cdn.com/content/v1/56ef2b8127d4bd622b7ea7c0/1458887991008-2YH9UIVRGS2FCGOKLKZV/BIZGIF02.gif?format=1000w",use_column_width=True)
    elif (predictions  == 1):
        return st.markdown('<h1 class="title">Text is a bout Entertainment</h1>', unsafe_allow_html=True),st.image("https://media.tenor.com/WuysZictURgAAAAC/big-mouth-watching-a-movie.gif",use_column_width=True)
    elif (predictions == 2):
        return st.markdown('<h1 class="title">Text is a bout Politics</h1>', unsafe_allow_html=True),st.image("https://www.grapheine.com/wp-content/uploads/2020/11/cover-elections-inde-V2-1.gif",use_column_width=True)
    elif (predictions == 3):
        return st.markdown('<h1 class="title">Text is a bout Sports</h1>', unsafe_allow_html=True),st.image("https://i.gifer.com/Apkz.gif",use_column_width=True)

    else:
        return st.markdown('<h1 class="title">Text is a bout Technology</h1>', unsafe_allow_html=True),st.image("https://media.kulfyapp.com/0NTUD7/0NTUD7-360.gif",use_column_width=True)
  
def set_bg_hack_url():
     '''
     A function to unpack an image from url and set as bg.
     Returns
     -------
     The background.
     '''
         
     st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://d1m75rqqgidzqn.cloudfront.net/wp-data/2020/07/29120707/iStock-1142254497.jpg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

# Custom title using HTML with the "title" class
st.markdown('<h1 class="title">Multi Class Text Classification On NEWS</h1>', unsafe_allow_html=True)

# Adjust the CSS style for the title
st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

input_text = st.text_input("Enter a Text here :thinking_face: :thinking_face:")






dig =""
if st.button("Detect type of text	:hugging_face:"):
    dig = analysis([input_text])

st.success(dig)















