import joblib
import numpy as np
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings("ignore")

# Function to load the saved model
def load_model():
    return joblib.load('cyclone_model.pkl')

# Function to predict the next 6 rows based on input
def predict_next_6_rows(lat_present, lon_present, dist2land_present, storm_speed_present,
                        year_present, month_present, day_present, hour_present,
                        lat_prev, lon_prev, dist2land_prev, storm_speed_prev):
    
    # Construct the current and previous data as feature arrays
    current_data = [lat_present, lon_present, dist2land_present, storm_speed_present, year_present, month_present, day_present, hour_present]
    previous_data = [lat_prev, lon_prev, dist2land_prev, storm_speed_prev, year_present, month_present, day_present, hour_present - 3]

    # Adjust for negative hours
    if previous_data[-1] < 0:
        previous_data[-1] += 24
        previous_data[6] -= 1  # Adjust day

    # Preprocess the input into the required shape (2 rows, 8 columns)
    input_data = [previous_data, current_data]
    input_data = np.array(input_data).reshape(1, 2, 8)
    
    # Flatten the input to match the model's input shape
    input_data_flat = input_data.reshape(1, -1)

    # Load the model and make predictions
    loaded_model = load_model()
    predictions = loaded_model.predict(input_data_flat)

    # Reshape the predictions back to (6, 4) format
    predictions_reshaped = predictions.reshape(6, 4)

    # Create a DataFrame for the predictions
    columns = ['LAT', 'LON', 'DIST2LAND', 'STORM_SPEED']
    df_predictions = pd.DataFrame(predictions_reshaped, columns=columns)

    # Add the 'Hour' column, incrementing by 3 hours from the present time
    df_predictions['Hour'] = [(hour_present + (i + 1) * 3) % 24 for i in range(6)]  # Ensure the hour wraps around 24

    # Display the DataFrame
    return df_predictions

# Function to plot predictions on a folium map and return the HTML representation
def plot_predictions_on_map(df_predictions):
    # Extract LAT and LON from the predictions
    latitudes = df_predictions['LAT'].tolist()
    longitudes = df_predictions['LON'].tolist()

    # Create a folium map centered at the first predicted point
    m = folium.Map(location=[latitudes[0], longitudes[0]], zoom_start=6)

    # Add the predicted points to the map and connect them with a polyline
    locations = list(zip(latitudes, longitudes))

    # Add the points to the map
    for lat, lon in locations:
        folium.Marker([lat, lon]).add_to(m)

    # Add a polyline to connect the points
    folium.PolyLine(locations, color='blue', weight=2.5, opacity=0.7).add_to(m)

    return m

# Streamlit App
def main():
    st.title("Cyclone Path Prediction")
    st.write("Input current and previous cyclone data to predict for the next 18 hours and visualize the path on a map.")

    # User inputs
    lat_present = st.number_input("Current Latitude", format="%f")
    lon_present = st.number_input("Current Longitude", format="%f")
    dist2land_present = st.number_input("Current DIST2LAND", format="%f")
    storm_speed_present = st.number_input("Current STORM_SPEED", format="%f")
    year_present = st.number_input("Current Year", format="%d")
    month_present = st.number_input("Current Month", format="%d")
    day_present = st.number_input("Current Day", format="%d")
    hour_present = st.number_input("Current Hour", format="%d")
    
    lat_prev = st.number_input("Previous Latitude", format="%f")
    lon_prev = st.number_input("Previous Longitude", format="%f")
    dist2land_prev = st.number_input("Previous DIST2LAND", format="%f")
    storm_speed_prev = st.number_input("Previous STORM_SPEED", format="%f")

    if st.button("Predict"):
        # Get the DataFrame prediction
        df_predictions = predict_next_6_rows(lat_present, lon_present, dist2land_present, storm_speed_present,
                                             year_present, month_present, day_present, hour_present,
                                             lat_prev, lon_prev, dist2land_prev, storm_speed_prev)
        
        # Display the predicted DataFrame
        st.write("Predicted DataFrame:")
        st.write(df_predictions)
        
        # Generate and display the map
        st.write("Cyclone Path Map:")
        map_ = plot_predictions_on_map(df_predictions)
        folium_static(map_)

if __name__ == "__main__":
    main()
