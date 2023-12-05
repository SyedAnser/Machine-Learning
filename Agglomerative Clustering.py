import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/syeda/OneDrive/Desktop/Datasets/CC GENERAL.csv")
data=data.dropna()

# Extract features
features = data.drop('CUST_ID', axis=1)  

# Check for NaN or infinite values in the data
if features.isnull().values.any() or not np.isfinite(features).all().all():
    raise ValueError("Dataset contains NaN or infinite values. Please handle missing values before proceeding.")

# Normalize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Check for NaN or infinite values after normalization
if np.isnan(features_scaled).any() or not np.isfinite(features_scaled).all():
    raise ValueError("Normalized data contains NaN or infinite values. Please handle missing values before proceeding.")

average_distance = np.mean(pdist(features_scaled))

# Check if the average distance is NaN or infinite
if not np.isfinite(average_distance):
    raise ValueError("Average distance is NaN or infinite. Please check your data for potential issues.")

# Agglomerative clustering with single linkage and threshold
model_single = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=average_distance)
clusters_single = model_single.fit_predict(features_scaled)
num_clusters_single = len(np.unique(clusters_single))

# Agglomerative clustering with complete linkage and threshold
model_complete = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=average_distance)
clusters_complete = model_complete.fit_predict(features_scaled)
num_clusters_complete = len(np.unique(clusters_complete))

# Agglomerative clustering with ward linkage and threshold
model_ward = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=average_distance)
clusters_ward = model_ward.fit_predict(features_scaled)
num_clusters_ward = len(np.unique(clusters_ward))

print(f'Number of Clusters (Single Linkage): {num_clusters_single}')
print(f'Number of Clusters (Complete Linkage): {num_clusters_complete}')
print(f'Number of Clusters (Ward Linkage): {num_clusters_ward}')

def calculate_cophenetic_correlation(Z, data):
    c, coph_dists = cophenet(Z, pdist(data))
    return c

coph_corr_single = calculate_cophenetic_correlation(linkage(features_scaled, method='single'), features_scaled)
coph_corr_complete = calculate_cophenetic_correlation(linkage(features_scaled, method='complete'), features_scaled)
coph_corr_ward = calculate_cophenetic_correlation(linkage(features_scaled, method='ward'), features_scaled)

print(f'Cophenetic correlation (Single Linkage): {coph_corr_single}')
print(f'Cophenetic correlation (Complete Linkage): {coph_corr_complete}')
print(f'Cophenetic correlation (Ward Linkage): {coph_corr_ward}')

def plot_dendrogram(model, **kwargs):
    Z = linkage(features_scaled, model.linkage)
    dendrogram(Z, **kwargs)

plt.figure(figsize=(6, 4))
plt.title('Dendrogram (Single Linkage)')
plot_dendrogram(model_single,truncate_mode='lastp', p=12)

plt.figure(figsize=(6, 4))
plt.title('Dendrogram (Complete Linkage)')
plot_dendrogram(model_complete,truncate_mode='lastp', p=12)

plt.figure(figsize=(6, 4))
plt.title('Dendrogram (Ward Linkage)')
plot_dendrogram(model_ward,truncate_mode='lastp', p=12)
plt.show()

