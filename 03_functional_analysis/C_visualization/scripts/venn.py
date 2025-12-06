import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import os

# --- CONFIGURATION: THE FULL RULES ---
# Copied exactly from your final_summary.txt
RULES = {
    "Metabolic\n(Rule 1)": [
        ("NCOR2", 0), ("FAM19A5", 0), ("BRSK2", 0), ("MT-ND4L", 1), 
        ("BIN1", 0), ("SEMA4D", 0), ("FAM102A", 0), ("ZFYVE16", 0), 
        ("ST3GAL4", 0), ("ATP6V1D", 0), ("DSTN", 0), ("CHST3", 0), 
        ("PALD1", 0), ("COX4I1", 0), ("FNTA", 0)
    ],
    "Iron/Stress\n(Rule 2)": [
        ("FTH1", 1), ("RBX1", 0), ("PALD1", 0), ("SLTM", 0), ("TMEM219", 0)
    ],
    "Regeneration\n(Rule 4)": [
        ("PALD1", 1)
    ]
}

def get_mask(df, conditions):
    mask = pd.Series(True, index=df.index)
    for gene, val in conditions:
        if gene in df.columns:
            mask = mask & (df[gene] == val)
    return mask

def plot_venn():
    print("Generating EXACT Overlap Venn Diagram...")
    
    # 1. Load Data
    # Try multiple paths
    if os.path.exists("./sc_data/X_binary.csv"):
        X = pd.read_csv("./sc_data/X_binary.csv")
        y = pd.read_csv("./sc_data/y_labels.csv")
    else:
        X = pd.read_csv("X_binary.csv")
        y = pd.read_csv("y_labels.csv")

    # 2. Filter for PD Only (Label = 1)
    pd_indices = y.iloc[:,0] == 1
    X_pd = X[pd_indices]
    print(f"Analyzing {len(X_pd)} PD cells...")

    # 3. Calculate Sets
    sets = {}
    for name, conditions in RULES.items():
        mask = get_mask(X_pd, conditions)
        sets[name] = set(X_pd.index[mask])
        print(f"  {name.replace(chr(10), ' ')}: {len(sets[name])} cells")

    # 4. Plot
    plt.figure(figsize=(8, 8))
    
    labels = list(RULES.keys())
    set_list = [sets[l] for l in labels]
    
    # Colors: Red, Orange, Blue
    out = venn3(set_list, set_labels=labels, set_colors=('#d62728', '#ff7f0e', '#1f77b4'), alpha=0.5)
    
    plt.title("Minimal Overlap between PD Failure Modes", fontsize=16, fontweight='bold')
    
    outfile = "subgroup_overlap_venn.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved to {outfile}")
    plt.show()

if __name__ == "__main__":
    plot_venn()
