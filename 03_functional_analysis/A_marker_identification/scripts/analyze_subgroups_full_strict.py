import pandas as pd
import numpy as np
import os

# --- CONFIGURATION: FULL RULES ---
RULES_TO_ANALYZE = [
    (
        "Rule 1 (Metabolic - STRICT)", 
        [
            ("NCOR2", 0), ("FAM19A5", 0), ("BRSK2", 0), ("MT-ND4L", 1), 
            ("BIN1", 0), ("SEMA4D", 0), ("FAM102A", 0), ("ZFYVE16", 0), 
            ("ST3GAL4", 0), ("ATP6V1D", 0), ("DSTN", 0), ("CHST3", 0), 
            ("PALD1", 0), ("COX4I1", 0), ("FNTA", 0)
        ]
    ),
    (
        "Rule 2 (Iron - STRICT)", 
        [("FTH1", 1), ("RBX1", 0), ("PALD1", 0), ("SLTM", 0), ("TMEM219", 0)]
    ),
    (
        "Rule 4 (Fibrotic - STRICT)", 
        [("PALD1", 1)] # Rule 4 was only 1 gene long anyway!
    )
]

# Difference threshold (keep at 5% to catch subtle signals in smaller groups)
ENRICHMENT_THRESHOLD = 0.05
MIN_SUBGROUP_SIZE = 10

def load_data():
    print("Loading data...")
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
        print("Error: Could not find X_binary.csv and y_labels.csv.")
        return None

    df = X.copy()
    target_col = y.columns[0] 
    df['target_label'] = y[target_col]
    return df

def get_subgroup_mask(df, rule_tuples):
    mask = pd.Series(True, index=df.index)
    for gene, val in rule_tuples:
        if gene not in df.columns:
            continue
        mask = mask & (df[gene] == val)
    return mask

def run_analysis_for_group(df, rule_name, rule_tuples, target_label, group_name_str):
    print(f"\n  --- Group: {group_name_str} (Label={target_label}) ---")
    
    group_mask = df['target_label'] == target_label
    df_group = df[group_mask].copy()
    
    if len(df_group) == 0:
        return

    subgroup_mask = get_subgroup_mask(df_group, rule_tuples)
    target_subgroup = df_group[subgroup_mask]
    background_group = df_group[~subgroup_mask]
    
    n_sub = len(target_subgroup)
    n_bg = len(background_group)
    
    print(f"    Subgroup Count: {n_sub}")
    
    if n_sub < MIN_SUBGROUP_SIZE:
        print(f"    Skipping: Subgroup too small ({n_sub} < {MIN_SUBGROUP_SIZE})")
        return

    results = []
    skip_genes = set([t[0] for t in rule_tuples] + ['target_label'])
    all_genes = [c for c in df_group.columns if c not in skip_genes]
    
    for gene in all_genes:
        freq_sub = target_subgroup[gene].mean()
        freq_bg = background_group[gene].mean()
        diff = freq_sub - freq_bg
        
        if abs(diff) > ENRICHMENT_THRESHOLD: 
            results.append({
                "Gene": gene,
                "Freq_In_Subgroup": freq_sub,
                "Freq_In_Rest": freq_bg,
                "Difference": diff,
                "Fold_Change": freq_sub / (freq_bg + 0.001)
            })
            
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("    No enriched genes found.")
        return

    results_df = results_df.sort_values(by="Difference", ascending=False)
    
    print("    Top 5 Enriched Genes:")
    print(results_df.head(5)[['Gene', 'Difference']].to_string(index=False))
    
    safe_rule = rule_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "pos").replace("/", "_").replace("-", "_")
    filename = f"enrichment_{safe_rule}_{group_name_str}.csv"
    results_df.to_csv(filename, index=False)
    print(f"    Saved: {filename}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        for name, rule in RULES_TO_ANALYZE:
            print(f"\n=== ANALYZING: {name} ===")
            run_analysis_for_group(df, name, rule, target_label=1, group_name_str="PD")
            run_analysis_for_group(df, name, rule, target_label=0, group_name_str="CTRL")
