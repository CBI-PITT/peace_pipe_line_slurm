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

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm='ball_tree',
        n_jobs=-1
    )

    labels = dbscan.fit_predict(points)
    print("Total clusters:", np.unique(labels).size)

    # Create DataFrame and use groupby (fastest for centroid calculation)
    temp_df = pd.DataFrame(points, columns=['axis-0', 'axis-1', 'axis-2'])
    temp_df['cluster'] = labels

    out_df = temp_df.groupby('cluster')[['axis-0', 'axis-1', 'axis-2']].mean().reset_index(drop=True)
    return out_df


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
