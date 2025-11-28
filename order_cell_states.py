import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# --- CONFIGURATION ---
# Rules as logic masks
RULES = {
    "1. Metabolic (Rule 1)": [("NCOR2", 0), ("MT-ND4L", 1)],
    "2. Iron-Toxic (Rule 2)": [("FTH1", 1), ("RBX1", 0)],
    "3. Fibrotic (Rule 4)": [("PALD1", 1)]
}

def get_mask(df, conditions):
    mask = pd.Series(True, index=df.index)
    for gene, val in conditions:
        if gene in df.columns:
            mask = mask & (df[gene] == val)
    return mask

def analyze_progression():
    print("Analyzing Disease Progression Trajectory...")
    
    # 1. Load Data
    try:
        X = pd.read_csv("./sc_data/X_binary.csv")
        y = pd.read_csv("./sc_data/y_labels.csv")
    except:
        X = pd.read_csv("X_binary.csv")
        y = pd.read_csv("y_labels.csv")

    # 2. Define the "Healthy Reference"
    # All cells from Control Patients (y=0)
    # OPTIONAL: Exclude cells that fall into your 'bad' subgroups to get "Super Healthy"
    ctrl_mask = y.iloc[:,0] == 0
    
    # Calculate the "Healthy Centroid" (The average profile of a healthy cell)
    # In binary data, this is the frequency vector (0.0 to 1.0) for every gene
    healthy_centroid = X[ctrl_mask].mean(axis=0).values.reshape(1, -1)
    
    results = []
    
    # 3. Analyze Each Subgroup (PD Cells Only)
    pd_mask = y.iloc[:,0] == 1
    
    # Also analyze "Background PD" (PD cells not in any rule)
    # This serves as a baseline for "General Disease"
    combined_rule_mask = pd.Series(False, index=X.index)
    for name, conditions in RULES.items():
        combined_rule_mask |= get_mask(X, conditions)
    
    bg_mask = pd_mask & (~combined_rule_mask)
    if bg_mask.sum() > 0:
        bg_cells = X[bg_mask].values
        # Calculate distances for every single cell to the healthy centroid
        dists = cdist(bg_cells, healthy_centroid, metric='cityblock') # Cityblock = Manhattan/Hamming-like
        for d in dists:
            results.append({"State": "Background PD", "Distance": d[0]})

    # Analyze specific rules
    for name, conditions in RULES.items():
        mask = get_mask(X, conditions) & pd_mask
        if mask.sum() == 0: continue
        
        cells = X[mask].values
        # Calculate distance of THESE cells to Healthy Centroid
        dists = cdist(cells, healthy_centroid, metric='cityblock')
        
        for d in dists:
            results.append({"State": name, "Distance": d[0]})

    # 4. Plot
    df_res = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    
    # Boxplot sorted by median distance
    # We want to see: Background < Rule 1 < Rule 2 < Rule 4?
    order = df_res.groupby("State")["Distance"].median().sort_values().index
    
    sns.boxplot(data=df_res, x="State", y="Distance", order=order, palette="viridis")
    
    plt.title("Trajectory of Failure: Transcriptomic Distance from Health", fontsize=14)
    plt.ylabel("Manhattan Distance from Healthy Centroid", fontsize=12)
    plt.xlabel("Cell State", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("trajectory_boxplot.png", dpi=300)
    print("Trajectory plot saved to trajectory_boxplot.png")
    
    # Print Stats
    print("\n--- Median Distances (Progression Order) ---")
    print(df_res.groupby("State")["Distance"].median().sort_values())

if __name__ == "__main__":
    analyze_progression()
