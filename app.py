import streamlit as st
import pandas as pd
import json
import os
from model.crop_yield_model import load_data, preprocess_data, load_model 

# Path to your dataset
data_path = 'data/dataset.csv'

# Load and preprocess the data
data = load_data(data_path)

with open('image_json_files.json', 'r') as f:
    crop_images = json.load(f)

# Function to get user input features
def user_input_features(data):
    crop = st.sidebar.selectbox('Select Crop', sorted(data['Crop'].unique()))
    season = st.sidebar.selectbox('Select Season', sorted(data['Season'].unique()))
    state = st.sidebar.selectbox('Select State', sorted(data['State'].unique()))
    area = st.sidebar.slider('Area', min_value=float(data['Area'].min()), max_value=float(data['Area'].max()),
                                   value=float(data['Area'].median()))
    production = st.sidebar.slider('Production', min_value=float(data['Production'].min()),
                                         max_value=float(data['Production'].max()),
                                         value=float(data['Production'].median()))
    annual_rainfall = st.sidebar.slider('Annual Rainfall', min_value=float(data['Annual_Rainfall'].min()),
                                              max_value=float(data['Annual_Rainfall'].max()),
                                              value=float(data['Annual_Rainfall'].median()))
    fertilizer = st.sidebar.slider('Fertilizer', min_value=float(data['Fertilizer'].min()),
                                         max_value=float(data['Fertilizer'].max()),
                                         value=float(data['Fertilizer'].median()))
    pesticide = st.sidebar.slider('Pesticide', min_value=float(data['Pesticide'].min()),
                                        max_value=float(data['Pesticide'].max()),
                                        value=float(data['Pesticide'].median()))
    user_data = {
        'Crop': crop,
        'Crop_Year': data['Crop_Year'].mode()[0],  # Use mode or current year
        'Season': season,
        'State': state,
        'Area': area,
        'Annual_Rainfall': annual_rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide,
        'Production': 0  # Placeholder for consistency
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Function to preprocess data for prediction
def preprocess_data_for_prediction(input_data, columns):
    input_data = pd.get_dummies(input_data, columns=['Crop', 'Season', 'State'])
    input_data = input_data.reindex(columns=columns, fill_value=0)
    return input_data

# Function to preprocess data for recommendations
def preprocess_data_for_recommendation(input_data):
    return input_data[['Season', 'State']]

# Get user input
input_data = user_input_features(data)


# Load the trained model
try:
    model, columns, mae, r2 = load_model('model/crop_yield_model.pkl')
    model_loaded = True
except FileNotFoundError:
    st.error("Model file not found. Please train the model first.")
    model_loaded = False
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    model_loaded = False

# Preprocess input data and make predictions
if model_loaded:
    input_data_preprocessed = preprocess_data_for_prediction(input_data, columns)
    
    
    # Ensure 'Yield' is not included in the prediction data
    if 'Yield' in input_data_preprocessed:
        input_data_preprocessed = input_data_preprocessed.drop(columns=['Yield'])
    
    # Make predictions if model loaded successfully
    if model_loaded:
        prediction = model.predict(input_data_preprocessed)
        predicted_yield = prediction[0] * 100  # Convert to percentage

        st.markdown('<style>body{text-align: center;}</style>', unsafe_allow_html=True)
        st.title('Crop Yield Prediction')
        st.subheader('Predicted Yield')
        st.write(f'{predicted_yield:.2f}%')
        # Display model evaluation metrics
        st.subheader('Model Evaluation Metrics')
        #st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        # Recommend best crop based on historical data
        try:
            # Filter data based on season and state from original input data
            input_data_recommendation = preprocess_data_for_recommendation(input_data)
            filtered_data = data[(data['Season'] == input_data_recommendation['Season'].values[0]) & (data['State'] == input_data_recommendation['State'].values[0])]
            
            # Debug: Display filtered data
            st.write("Filtered Data:", filtered_data)
            
            if filtered_data.empty:
                st.error("No historical data available for the selected Season and State.")
            else:
                avg_yields = filtered_data.groupby('Crop')['Yield'].mean()
                recommended_crop = avg_yields.idxmax()
                st.subheader('Recommended Crop for Maximum Yield and Profit')
                st.write(recommended_crop)
                # Display the image of the recommended crop
                if recommended_crop in crop_images:
                  crop_image_path = crop_images[recommended_crop]
                  #st.markdown(f'<div style="text-align: center;"><img src="{crop_image_path}" width="500" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px; box-shadow: 0px 0px 5px 0px #888;"></div>', unsafe_allow_html=True)
                  #st.image(crop_image_path, width=500 )
                  st.image(crop_image_path, caption=f'Image of {recommended_crop}', use_column_width=True,width=700)
                else:
                  st.warning("Image for the recommended crop not found.")
                    
        except KeyError as e:
            st.error(f"Key error: {e}. This might be due to missing columns in the input data.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
