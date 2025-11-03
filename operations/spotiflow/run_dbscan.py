import os
import sys
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def run_dbscan_on_df(df, eps, min_samples):
    points = df[["axis-0", "axis-1", "axis-2"]].to_numpy()
    print("Points shape", points.shape)

    points = points[~np.any(np.isnan(points), axis=1)]
    if points.shape[0] == 0:
        return pd.DataFrame()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(points)

    # get the cluster assignments for each point
    labels = dbscan.labels_

    # get the unique cluster labels
    cluster_labels = np.unique(labels)
    print("Total clusters:", len(cluster_labels))

    # calculate the centroid of each cluster
    def get_cluster_centroid(label):
        points_in_cluster = points[labels == label]
        centroid = np.mean(points_in_cluster, axis=0)
        return centroid

    centroids = [dask.delayed(get_cluster_centroid)(x) for x in cluster_labels]
    cluster_centroids = dask.compute(*centroids)
    cluster_centroids = np.array(cluster_centroids)
    # for label in cluster_labels:
    #     print("label", label)
    #     # get the points in the current cluster
    #     points_in_cluster = points[labels == label]
    #
    #     # calculate the mean of the points in the cluster to get the centroid
    #     centroid = np.mean(points_in_cluster, axis=0)
    #
    #     # add the centroid to the list of cluster centroids
    #     cluster_centroids.append(centroid)
    #
    # cluster_centroids = np.array(cluster_centroids)

    out_df = pd.DataFrame()
    out_df['axis-0'] = cluster_centroids[:, 0]
    out_df['axis-1'] = cluster_centroids[:, 1]
    out_df['axis-2'] = cluster_centroids[:, 2]
    return out_df



print("Doing DBSCAN")
input_file = sys.argv[1]
output_dir = sys.argv[2]
model_name = sys.argv[3]
eps = 10
min_samples = 1

print("input file", input_file)
print("reading df")
df = pd.read_csv(input_file)
print("Read df")

df = run_dbscan_on_df(df, eps, min_samples)
print("Processed df. Saving output")
df.to_csv(os.path.join(output_dir, f"{model_name}_merged_dbscan_df.csv"))
