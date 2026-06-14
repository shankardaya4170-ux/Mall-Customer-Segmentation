from evaluation import model_acc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import joblib



def clustering(data):
    cluster_data=data[["Annual Income (k$)", "Spending Score (1-100)"]].copy()
    print(cluster_data.head())


    wcss=[]
    for i in range (1,11):
        kmeans=KMeans(n_clusters=i,init='k-means++', random_state=42)
        kmeans.fit(cluster_data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 11), wcss, marker="o", linestyle="--", color="purple")
    plt.title("The Elbow Method (Optimal K Dhoondhna)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.grid(True)
    plt.show()
    return cluster_data


def final_clustering(cluster_data, original_data):
    final_kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_predict = final_kmeans.fit_predict(cluster_data)
    original_data['Cluster_Number'] = y_predict
    print("Success: Customers 5 alag-alag clusters mein bant gaye hain!")
    print(original_data.head())

    model_acc(cluster_data, y_predict)
    # --- NAYA CODE: MODEL KO SAVE KARNA ---
    joblib.dump(final_kmeans, 'kmeans_model.pkl')

    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster_Number', data=original_data,
                    palette='bright', s=100)
    plt.show()
    return original_data