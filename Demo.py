# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 03:45:46 2023

@author: Rohan.Mitra
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from prophet.serialize import model_to_json, model_from_json

st.set_page_config(layout='wide')

st.title('Revenue Prediction Using Google and Facebook Ad Spend')
st.header("Upload a file in the specified format and recieve predictions for the expected revenue generated")

st.divider()

data=pd.DataFrame({'Date':[f'2023-4-{i}' for i in range(1,31)],
                   'FCost':np.random.randint(0,1000,(30,)),
                   'GCost':np.random.randint(0,1000,(30,))})

st.subheader("Specified Data Format:")
st.write(data)


##################### DATA UPLOAD  ##########################

st.header('Easy to use! Just upload your data, click RUN and it will present the prediction')

uploaded_file = st.file_uploader('Upload your data in the specified format')     #Use uploaded_file variable for anything
if uploaded_file:
    data=pd.read_csv(uploaded_file)
    st.subheader("Recieved & preprocessed data:")
    #st.write(data)
    # Prepare the file in the format needed:
    data['TCost']=data['FCost']+data['GCost']
    import datetime
    def weekOfYear(x):
        return int(datetime.date(*list(map(int,x.split('-')))).strftime('%W')) 
    	#return int(datetime.date(*list(map(int,x.split('-')))).strftime('%W'))
    
    def weekOfMonth(x):
        y,m,d=x.split('-')
        return weekOfYear(x)-weekOfYear(y+'-'+m+'-01')
        #return weekOfYear(x)-weekOfYear(y+r'/'+m+r'/01')
    
    data['weekOfYear']=data['Date'].apply(weekOfYear)
    data['weekOfMonth']=data['Date'].apply(weekOfMonth)
    data.columns=['ds']+list(data.columns)[1:]
    st.write(data)
    
    
##################### DISPLAY OBJECT DETECTION RESULTS ##########################

    if st.button('Run'):    
        
        st.header("Predicting...")
        with open('serialized_model_AED_all_data.json', 'r') as fin:
            model = model_from_json(fin.read())  # Load model
            
        preds=model.predict(data)
        res=preds[['ds','yhat']]
        res.columns=['Date','Prediction']
        st.header("Results:")
        st.write(res)

        if st.download_button('Download Results', res.to_csv(index=False).encode('utf-8'), file_name='predictions.csv',mime='text/csv'):
            st.header("Downloaded successfully!")