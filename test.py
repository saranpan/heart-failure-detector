from data import import_dataset
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from pickle import load
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import base64
import plotly as px

#import model # trained on shorturl.at/gyAR1
pickled_model = load(open('model.pkl', 'rb'))

#import the data (we once used)
import_dataset('https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data')


# Streamlit
st.set_page_config(page_title='Hobot',page_icon=':man:',
                    layout="wide",
                    menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     })

@st.experimental_memo
def get_data():
    df = pd.read_csv('heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
    return df

@st.experimental_memo
def get_meta_data():
    mdf = pd.read_csv('metadata.csv')
    return mdf

def html_reader(html_file):
    HtmlFile = open(html_file, 'r', encoding='utf-8')
    page = HtmlFile.read() 
    components.html(page,scrolling=False)
    

df = get_data()
mdf = get_meta_data()
print(df)
map_df = df.replace({'sex':{0:'Female',1:'Male'}})

print(map_df)