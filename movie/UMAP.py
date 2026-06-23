import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import umap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_temp_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Split into the x-block and the y-value
            *x_block, y_str = line.split()
            x_str = x_block[0]  # the part like "0.095646_0.071834_..."

            # Convert x and y to numeric values
            x_vals = [float(v) for v in x_str.split('_')]
            y_val = float(y_str)

            data.append((x_vals, y_val))

    return data

# Example usage
xy_data = read_temp_file("temp.txt")
segment1 = xy_data[0:64]        # lines 1–78
segment2 = xy_data[64:2384]     # lines 79–2448
segment3 = xy_data[2448:] 

# Find the entry in segment1 with the minimal y value
min_x, min_y = min(segment1, key=lambda item: item[1])

print("Minimum y in segment1:", min_y)
print("Corresponding x:", min_x)

batch_size = 64
set_size = batch_size //16

# Step 1 — split segment2 into batches of 64
batches = [
    segment2[i:i + batch_size]
    for i in range(0, len(segment2), batch_size)
]

# Step 2 — split each batch into 4 sets of 16
batches_with_sets = []
for batch in batches:
    sets = [
        batch[j:j + set_size]
        for j in range(0, len(batch), set_size)
    ]
    batches_with_sets.append(sets)

# Example: examine batch 0
print(f"Number of batches: {len(batches_with_sets)}")
print(f"Sets in batch 0: {len(batches_with_sets[0])}")
print(f"Size of set 0 in batch 0: {len(batches_with_sets[0][0])}")

# batches_with_sets: list of batches, each batch is a list of 4 sets
min_y_info = []  # will store (batch_index, set_index, min_y, corresponding_x)

for b_idx, batch in enumerate(batches_with_sets):
    batch_min_y = float('inf')
    batch_min_x = None
    batch_min_set_idx = None

    for s_idx, set_ in enumerate(batch):
        # Find minimal y in this set
        set_min_x, set_min_y = min(set_, key=lambda item: item[1])

        if set_min_y < batch_min_y:
            batch_min_y = set_min_y
            batch_min_x = set_min_x
            batch_min_set_idx = s_idx

    min_y_info.append((b_idx, batch_min_set_idx, batch_min_y, batch_min_x))

# Example: print results
for info in min_y_info:
    b_idx, s_idx, min_y, x_vals = info
    print(f"Batch {b_idx}, Set {s_idx}: min_y = {min_y}, corresponding x = {x_vals}")

import matplotlib.pyplot as plt

initial_min_x, initial_min_y = min(segment1, key=lambda item: item[1])
arranged_segment2 = [item for batch in batches_with_sets for set_ in batch for item in set_]
parallel_points = 4

y_values = []  # running best y
last_x_values = []
current_best = initial_min_y

current_x = initial_min_x[-1] if hasattr(initial_min_x, '__len__') else initial_min_x

for i in range(0, len(arranged_segment2), parallel_points):
   
    iteration_points = arranged_segment2[i:i+parallel_points]
    
    min_point_in_iteration = min(iteration_points, key=lambda item: item[1])
    min_iter_x, min_iter_y = min_point_in_iteration

    if min_iter_y < current_best:
        current_best = min_iter_y

        current_x = min_iter_x[-1] if hasattr(min_iter_x, '__len__') else min_iter_x
    
    y_values.append(current_best)
    last_x_values.append(current_x) 


iterations = list(range(1, len(y_values)+1))
iterations = [0] + iterations
y_values = [initial_min_y] + y_values

initial_x_last = initial_min_x[-1] if hasattr(initial_min_x, '__len__') else initial_min_x
last_x_values = [initial_x_last] + last_x_values

plt.figure(figsize=(8,5))
plt.plot(iterations, y_values, marker='o', linestyle='-', label="Current Best y")
plt.plot(iterations, last_x_values, marker='s', linestyle='--', label="Last element of x", color='orange')
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Trust-Region BO: Running Best y vs Iteration (x last element)")
plt.legend()
plt.grid(True)
plt.show()

X = np.array([item[0] for item in segment2])  # shape (T, d)
y = np.array([item[1] for item in segment2])  # shape (T,)
T, d = X.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_tr = 4

X_tr = []
y_tr = []

for tr in range(n_tr):
    X_tr.append(X_scaled[tr::n_tr])
    y_tr.append(y[tr::n_tr])

k = 20
R_t_tr = []

for tr in range(n_tr):
    X_sub = X_tr[tr]
    T_sub = len(X_sub)
    R_list = []
    
    for t in range(T_sub):
        start = max(0, t-k+1)
        window = X_sub[start:t+1]
        
        center = np.mean(window, axis=0)
        distances = np.linalg.norm(window - center, axis=1)
        
        R_list.append(np.mean(distances))
    
    R_t_tr.append(np.array(R_list))

X = np.array([item[0] for item in segment2])

import os
import numpy as np
import matplotlib.pyplot as plt
import umap

start_idx = 0
end_idx = 2304
batch_size = 32

output_dir = "umap_progress_plots"
os.makedirs(output_dir, exist_ok=True)

output_dir = "umap_progress_plots"
os.makedirs(output_dir, exist_ok=True)

X_all = X[start_idx:end_idx+1]
n_samples = X_all.shape[0]

n_neighbors = min(20, n_samples - 1)

reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.1,
    random_state=42
)

embedding = reducer.fit_transform(X_all)

for left in range(0, n_samples, batch_size):

    right = min(left + batch_size, n_samples)

    plt.figure(figsize=(8,6))

    plt.scatter(
        embedding[:,0],
        embedding[:,1],
        s=18,
        c='lightgray',
        alpha=0.4
    )

    if left > 0:
        history_idx = np.arange(0, left)
        plt.scatter(
            embedding[history_idx,0],
            embedding[history_idx,1],
            s=25,
            c='blue',
            alpha=0.7,
            label="Previous samples"
        )

    new_idx = np.arange(left, right)
    plt.scatter(
        embedding[new_idx,0],
        embedding[new_idx,1],
        s=35,
        c='red',
        alpha=0.9,
        label="New samples"
    )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"Sampling step {right}/{n_samples}")

    plt.legend()
    plt.tight_layout()

    filename = f"sampling_{right:04d}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

import os
import imageio

def extract_number(fname):
 
    return int(fname.split('_')[1].split('.')[0])

img_folder = "umap_progress_plots"
images = []
for fname in sorted(os.listdir(img_folder), key=extract_number):
    if fname.endswith(".png"):
        images.append(imageio.imread(os.path.join(img_folder, fname)))

imageio.mimsave("evolution.mp4", images, fps=2)
