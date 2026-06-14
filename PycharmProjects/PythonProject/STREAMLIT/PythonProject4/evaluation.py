from sklearn.metrics import silhouette_score
def model_acc(cluster_data,prediction_data):
    score= silhouette_score(cluster_data,prediction_data)
    print(score)
    return score
