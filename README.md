# 🛍️ Mall Customer Segmentation Dashboard

## Overview
This is an end-to-end Machine Learning project that segments supermarket customers into distinct groups based on their purchasing behavior. It uses the **K-Means Clustering** algorithm (Unsupervised Learning) to identify patterns in 'Annual Income' and 'Spending Score'.

## Features
* **Optimal Clustering:** Calculates the ideal number of customer segments using the **Elbow Method** (WCSS).
* **Machine Learning:** Groups customers into 5 distinct categories using the Scikit-Learn K-Means model.
* **Interactive Dashboard:** Built a live frontend web app using **Streamlit**.
* **Live Visualizations:** Displays real-time Matplotlib plots (Elbow Graph & Scatter Plots with Centroids) directly on the web app.
* **Real-time Prediction Engine:** Users can input their own income and spending score via sliders to instantly find out which cluster they belong to.

## Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Data Manipulation & Math:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Web Framework:** Streamlit
* **Model Serialization:** Joblib

## How to Run This Project on Your Machine
1. Clone this repository.
2. Open your terminal and install the required libraries:
   ```bash
   pip install -r requirements.txt
