import pandas as pd
import numpy as np
import os

# --- TARGET GENE LISTS (From your Abstract) ---
TARGETS = {
    "Dopamine Metabolism": [
        "TH", "DDC", "SLC6A3", "SLC18A2", "DRD2", "MAOA", "MAOB", "COMT", "ALDH1A1"
    ],
    "MAPK Signaling": [
        "MAPK1", "MAPK3", "MAP2K1", "RAF1", "KRAS", "HRAS", "NRAS", "FOS", "JUN"
    ],
    "Renin-Angiotensin (RAS)": [
        "AGT", "REN", "ACE", "ACE2", "AGTR1", "AGTR2", "CTSA", "CTSB"
    ],
    "Oxidative Stress (New Hypothesis)": [
        "SOD1", "SOD2", "GPX1", "GPX4", "CAT", "GSR"
    ]
}

# --- SUBGROUPS ---
RULES = [
    ("Rule 1 (Metabolic)", [("NCOR2", 0), ("MT-ND4L", 1)]),
    ("Rule 2 (Iron)", [("FTH1", 1)]),
    ("Rule 4 (Fibrotic)", [("PALD1", 1)])
]

def check_targets():
    print("Checking Abstract Hypotheses against Discovery...")
    
    # Load Data
    if os.path.exists("./sc_data/X_binary.csv"):
        X = pd.read_csv("./sc_data/X_binary.csv")
        y = pd.read_csv("./sc_data/y_labels.csv")
    elif os.path.exists("X_binary.csv"):
        X = pd.read_csv("X_binary.csv")
        y = pd.read_csv("y_labels.csv")
    else:
        print("Data not found.")
        return

    # Filter for PD cells only
    X['target_label'] = y.iloc[:,0]
    df_pd = X[X['target_label'] == 1]
    
    for rule_name, conditions in RULES:
        print(f"\n=== {rule_name} ===")
        
        # Get Subgroup Mask
        mask = pd.Series(True, index=df_pd.index)
        for gene, val in conditions:
            if gene in df_pd.columns:
                mask = mask & (df_pd[gene] == val)
        
        subgroup = df_pd[mask]
        background = df_pd[~mask]
        
        # Check each hypothesis
        for pathway, genes in TARGETS.items():
            print(f"  Checking {pathway}...")
            hits = []
            for gene in genes:
                if gene in df_pd.columns:
                    freq_sub = subgroup[gene].mean()
                    freq_bg = background[gene].mean()
                    diff = freq_sub - freq_bg
                    
                    # If significantly different (> 5%)
                    if abs(diff) > 0.05:
                        direction = "+" if diff > 0 else "-"
                        hits.append(f"{gene} ({direction}{diff:.2f})")
            
            if hits:
                print(f"    -> FOUND: {', '.join(hits)}")
            else:
                print(f"    -> No significant signal.")

if __name__ == "__main__":
    check_targets()
