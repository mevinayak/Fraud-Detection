import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit.components.v1 as components

# Increase maximum file size to 500 MB
# st.set_option('server.maxUploadSize', 500)
# adding it in command line
st.set_page_config(page_title='Random Forest Classifier App',layout='wide')
# Define the Streamlit app
def app():
    #CSS
    custom_css = """
    <style>
        .fullscreen {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            background-color: #f0f2f6;
        }
        h1, h2, h3, h4, h5, h6, label,button,title{
            font-family: "Copperplate", Copperplate;
            color: #00FF33;
            margin-top: 100;
        }
        .stButton>button {
            background-color: #1a202c;
            color: #f0f2f6;
            font-weight: bold;
        }
    </style>
    """
    # Set page title
    image = Image.open("Banking-logo-by-Friendesign-Acongraphic-10-580x386.jpg")
    # Resize the image
    new_width = 400  # You can change this value to your preferred width
    new_height = int(new_width * image.height / image.width)
    image = image.resize((new_width, new_height))
    col1, col2, col3 = st.columns([8.8, 9, 8.8])
    with col2:
        st.image(image)
    #st.image(image)
    
    #st.image(image, use_column_width=True)
    
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown('<div class="fullscreen"></div>', unsafe_allow_html=True)
    with st.container():
        # Display title and description
        col1, col2 = st.columns([1.5, 3])
        with col2:
            st.title('Fraud Detection')
        
        st.markdown('Upload a CSV file with the input data and click "Run Inference" to predict the target variable using a Random Forest Classifier.')
        st.write('Upload a CSV file:')
        uploaded_file = st.file_uploader('', type='csv')
    # Display title and description
    #st.title('Random Forest Classifier App')
    # Display file upload widget
    ##st.write('Upload a CSV file:')
    #uploaded_file = st.file_uploader('', type='csv')
    
    fnames = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
    # Train the Random Forest Classifier when user clicks "Train Model"
    #button allignment 
    col1, col2= st.columns([7.15,9])
    with col2:
        run_TrainModel_button=st.button('Train Model')
    if run_TrainModel_button:
        # Check if file was uploaded
        if uploaded_file is not None:
            # Load the uploaded file into a dataframe
            df = pd.read_csv(uploaded_file)

            # Check if the dataframe has the required columns
            if set(fnames).issubset(set(df.columns)):
                
                # Train a Random Forest Classifier on the dataframe
                df = preprocess(df) # Do some preprocessing
                
                x_train,x_test,y_train,y_test = train_test(df=df)

                pred, score = RF_Classifier(x_train,x_test,y_train,y_test)
                

                st.success('Model trained.')
            else:
                st.error("CSV file must contain columns 'step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'")
        else:
            st.error('Please upload a CSV file.')
    col1, col2= st.columns([7,9])
    with col2:
        run_inference_button=st.button('Run Inference')
    # Run inference when user clicks "Run Inference"
    if run_inference_button and 'clf' in locals():
        # Check if file was uploaded
        if uploaded_file is not None:
            
            # Load the uploaded file into a dataframe
            df = pd.read_csv(uploaded_file)

            df = preprocess(df) # Do some preprocessing
            
            # Check if the dataframe has the required columns
            if set(fnames).issubset(set(df.columns)):
                # Run inference and display the results
                #y_pred = clf.predict(df[['feature_1', 'feature_2']])
                #accuracy = accuracy_score(df['target'], y_pred)
                #st.write(f'Predicted target values: {y_pred}')
                accuracy = 1.0
                st.write(f'Accuracy: {score}')
                st.write(f'preds: {st.dataframe(pred)}')
            else:
                st.error('CSV file must contain columns "feature_1" and "feature_2".')
        else:
            st.error('Please upload a CSV file.')


def preprocess(df):

    df['type']=df['type'].replace({'CASH_OUT':1,'PAYMENT':2,'CASH_IN':3,'TRANSFER':4,'DEBIT':5})
    return df


def train_test(df):

    X=df[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
    y=df[['isFraud']]

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

    return x_train,x_test,y_train,y_test


def RF_Classifier(x_train,x_test,y_train,y_test):

    rf=RandomForestClassifier()

    rf_model=rf.fit(x_train,np.ravel(y_train))
    score = rf_model.score(x_test,y_test)
    pred = rf_model.predict(x_test)

    return pred, score

if __name__ == "__main__":
    
    app()
    