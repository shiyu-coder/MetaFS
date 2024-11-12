from sklearn.cluster import KMeans
import numpy as np


def feature_space_reduction_clustering(corr_mat, N, n_clusters):
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)

    # Calculate the number of features per cluster
    n = N // n_clusters

    # Perform clustering
    labels = kmeans.fit_predict(corr_mat)

    # Post-process the clustering results to ensure each cluster contains exactly n features
    class_counts = np.bincount(labels)
    iter_count = 0

    while (not np.all(class_counts == n)) and iter_count < 5000:
        for i in range(n_clusters):
            if class_counts[i] > n:
                # If cluster i has more than n features, redistribute the excess features to other clusters
                excess_features = np.where(labels == i)[0][n:]
                distances = kmeans.transform(corr_mat[excess_features])
                nearest_classes = np.argsort(distances, axis=1)[:, 1:]

                for j, feature in enumerate(excess_features):
                    for cls in nearest_classes[j]:
                        if class_counts[cls] < n:
                            labels[feature] = cls
                            class_counts[i] -= 1
                            class_counts[cls] += 1
                            break
            elif class_counts[i] < n:
                # If cluster i has less than n features, borrow features from other clusters
                deficit = n - class_counts[i]
                mask = labels != i
                distances = kmeans.transform(corr_mat[mask])[:, i]
                nearest_features = np.argsort(distances)[:deficit]
                labels[np.where(mask)[0][nearest_features]] = i
                class_counts[i] += deficit
                class_counts[labels[mask][nearest_features]] -= 1

            class_counts = np.bincount(labels)
        iter_count += 1

    return labels

