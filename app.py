import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
pipe=pickle.load(open('pipe1.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
st.title('Laptop Price Predictor')

company=st.selectbox('Brand',df['Company'].unique())

TypeName=st.selectbox('Type',df['TypeName'].unique())

ram=st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])

weight=st.number_input('Weight')

touchscreen=st.selectbox('TouchScreen',['No','Yes'])

screen_size=st.number_input('Screen Size')

resolution=st.selectbox('Screen Resoltion',['1920x1080','1366x768','1600x900',
'3840x2160','3200x1800','2800x1800','2560x1800','2560x1600','2560x1440','2304x1440','1680x1050'])

cpu=st.selectbox('Cpu',df['Cpu Brand'].unique())

gpu=st.selectbox('Gpu',df['Gpu Brand'].unique())

os=st.selectbox("Os",df['Op Name'].unique())

if st.button('Predict'):
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res)**2)**0.5/screen_size
    dict = {'Company': {0:company}, "TypeName": {0:TypeName}, "Ram": {0:ram}, "Weight": {0:weight},
            "TouchScreen": {0:touchscreen}, "ppi": {0:ppi}, "Cpu Brand": {0:cpu},
            "Gpu Brand": {0:gpu}, "Op Name": {0:os}}
    d=pd.DataFrame(dict)
    st.title(np.exp(pipe.predict(d)[0]))



