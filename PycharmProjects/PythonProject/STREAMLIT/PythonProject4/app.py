import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
import numpy as np

# 1. Page Configuration (Wide layout for Dashboard)
st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Mall Customer Segmentation Dashboard")
st.markdown("---")

# 2. Data aur Model Load Karna
data = pd.read_csv('Mall_Customers.csv')  # Aapki original file
cluster_data = data.iloc[:, [3, 4]].values  # Income aur Spending Score
model = joblib.load('kmeans_model.pkl')

# 3. Dashboard ke liye 2 Columns banana
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Elbow Method Graph")
    # WCSS calculate karke Elbow graph banana
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(cluster_data)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), wcss, marker='o', color='red')
    ax1.set_title('Elbow Graph (Optimal K=5)')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS')
    st.pyplot(fig1)

with col2:
    st.subheader("🎯 Final Customer Clusters")
    # Naya model train karke Scatter plot banana
    final_kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_predict = final_kmeans.fit_predict(cluster_data)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    for i in range(5):
        ax2.scatter(cluster_data[y_predict == i, 0], cluster_data[y_predict == i, 1], s=50, c=colors[i],
                    label=f'Cluster {i + 1}')

    # Centroids
    ax2.scatter(final_kmeans.cluster_centers_[:, 0], final_kmeans.cluster_centers_[:, 1], s=200, c='yellow', marker='*',
                label='Centroids')
    ax2.set_xlabel('Annual Income (k$)')
    ax2.set_ylabel('Spending Score (1-100)')
    st.pyplot(fig2)

st.markdown("---")

# 4. User Prediction Section
st.subheader("🤖 Predict Your Cluster")
col3, col4 = st.columns(2)

with col3:
    income = st.slider("Annual Income (in k$)", min_value=10, max_value=150, value=50)
with col4:
    score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict My Cluster 🚀"):
    # Numpy array ki jagah Pandas DataFrame use kar rahe hain taaki naam match ho jayein
    input_data = pd.DataFrame([[income, score]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])

    predicted_cluster = model.predict(input_data)[0]
    st.success(f"🎉 Aap **Cluster {predicted_cluster + 1}** ke member hain!")