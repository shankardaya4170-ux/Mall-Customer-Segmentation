import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_loader import clustering
from data_loader import clustering, final_clustering


data = pd.read_csv("Mall_Customers.csv")  # we are load the dataset
print(data.head())  # read data first five row

print(data.isnull().sum())  # check the null value
print(data.duplicated().sum())  # check duplicate value

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(data["Age"], kde=True, color="skyblue")
plt.title("Age distribution")

plt.subplot(1,3,2)
sns.histplot(data["Annual Income (k$)"],kde=True,color="red")
plt.title("Annual Income")

plt.subplot(1,3,3)
sns.histplot(data["Spending Score (1-100)"],kde=True,color="red")
plt.title("Spending Score (1-100) ")

plt.tight_layout()
plt.show()

cluster_data = clustering(data)
final_data = final_clustering(cluster_data, data)

