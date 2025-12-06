import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import os

# --- CONFIGURATION: FULL RULES ---
# Copied from your results file
RULES = {
    "1. Metabolic\n(Rule 1)": [
        ("NCOR2", 0), ("FAM19A5", 0), ("BRSK2", 0), ("MT-ND4L", 1), 
        ("BIN1", 0), ("SEMA4D", 0), ("FAM102A", 0), ("ZFYVE16", 0), 
        ("ST3GAL4", 0), ("ATP6V1D", 0), ("DSTN", 0), ("CHST3", 0), 
        ("PALD1", 0), ("COX4I1", 0), ("FNTA", 0)
    ],
    "2. Iron-Toxic\n(Rule 2)": [
        ("FTH1", 1), ("RBX1", 0), ("PALD1", 0), ("SLTM", 0), ("TMEM219", 0)
    ],
    "3. Fibrotic\n(Rule 4)": [
        ("PALD1", 1)
    ]
}

def get_mask(df, conditions):
    mask = pd.Series(True, index=df.index)
    for gene, val in conditions:
        if gene in df.columns:
            mask = mask & (df[gene] == val)
    return mask

def analyze_progression():
    print("Analyzing Disease Progression Trajectory (STRICT RULES)...")
    
    # 1. Load Data
    paths_to_try = [
        ("./X_binary.csv", "./y_labels.csv"),
        ("./sc_data/X_binary.csv", "./sc_data/y_labels.csv")
    ]
    X, y = None, None
    for x_path, y_path in paths_to_try:
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"  Found files at: {x_path}")
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            break

    if X is None:
        print("Error: Could not find data files.")
        return

    # 2. Define "Healthy Reference"
    ctrl_mask = y.iloc[:,0] == 0
    healthy_centroid = X[ctrl_mask].mean(axis=0).values.reshape(1, -1)
    
    results = []
    
    # 3. Analyze Subgroups (PD Only)
    pd_mask = y.iloc[:,0] == 1
    
    # Calculate Background PD (PD cells NOT in any strict rule)
    combined_rule_mask = pd.Series(False, index=X.index)
    for name, conditions in RULES.items():
        combined_rule_mask |= get_mask(X, conditions)
    
    bg_mask = pd_mask & (~combined_rule_mask)
    if bg_mask.sum() > 0:
        bg_cells = X[bg_mask].values
        dists = cdist(bg_cells, healthy_centroid, metric='cityblock')
        for d in dists:
            results.append({"State": "Background PD", "Distance": d[0]})

    # Calculate Specific Rules
    for name, conditions in RULES.items():
        mask = get_mask(X, conditions) & pd_mask
        count = mask.sum()
        print(f"  {name.replace(chr(10), ' ')}: {count} PD cells")
        
        if count == 0: continue
        
        cells = X[mask].values
        dists = cdist(cells, healthy_centroid, metric='cityblock')
        for d in dists:
            results.append({"State": name, "Distance": d[0]})

    # 4. Plot
    df_res = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    
    # Sort by median distance to show trajectory
    order = df_res.groupby("State")["Distance"].median().sort_values().index
    
    sns.boxplot(data=df_res, x="State", y="Distance", order=order, palette="viridis")
    
    plt.title("Trajectory of Failure: Transcriptomic Distance from Health (Strict Definitions)", fontsize=14)
    plt.ylabel("Manhattan Distance from Healthy Centroid", fontsize=12)
    plt.xlabel("Cell State", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("trajectory_boxplot.png", dpi=300)
    print("Trajectory plot saved to trajectory_boxplot.png")
    
    # Print Stats for Paper
    print("\n--- Median Distances (Progression Order) ---")
    print(df_res.groupby("State")["Distance"].median().sort_values())

if __name__ == "__main__":
    analyze_progression()
