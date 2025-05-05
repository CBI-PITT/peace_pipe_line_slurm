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


def run_dbscan_on_df(df, eps, min_samples):
    points = df[["axis-0", "axis-1", "axis-2"]].to_numpy()
    print("Points shape", points.shape)

    points = points[~np.any(np.isnan(points), axis=1)]

    # create a DBSCAN object
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # fit the points to the model
    dbscan.fit(points)

    # get the cluster assignments for each point
    labels = dbscan.labels_

    # get the unique cluster labels
    cluster_labels = np.unique(labels)
    print("Total clusters:", len(cluster_labels))

    # calculate the centroid of each cluster
    cluster_centroids = []

    # def get_cluster_centroid(label):
    #     points_in_cluster = points[labels == label]
    #     centroid = np.mean(points_in_cluster, axis=0)
    #     return centroid
    #
    # centroids = [dask.delayed(get_cluster_centroid)(x) for x in cluster_labels]
    # cluster_centroids = dask.compute(centroids)[0]

    for label in cluster_labels:
        print("label", label)
        # get the points in the current cluster
        points_in_cluster = points[labels == label]

        # calculate the mean of the points in the cluster to get the centroid
        centroid = np.mean(points_in_cluster, axis=0)

        # add the centroid to the list of cluster centroids
        cluster_centroids.append(centroid)

    cluster_centroids = np.array(cluster_centroids)

    df = pd.DataFrame()
    df['axis-0'] = cluster_centroids[:, 0]
    df['axis-1'] = cluster_centroids[:, 1]
    df['axis-2'] = cluster_centroids[:, 2]
    return df

print("Doing DBSCAN")
input_file = sys.argv[1]
output_dir = sys.argv[2]
eps = int(sys.argv[3])
min_samples = int(sys.argv[4])

print("input file", input_file)
print("reading df")
df = pd.read_csv(input_file)
print("Read df")
df = run_dbscan_on_df(df, eps, min_samples)
print("Processed df. Saving output")
df.to_csv(os.path.join(output_dir, f"dbscan_{os.path.basename(input_file)}"), index=False)
