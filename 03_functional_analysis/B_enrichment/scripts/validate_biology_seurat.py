import pandas as pd
import gseapy as gp
import os

# --- CONFIGURATION ---
# Map Rule Names to the MAST output files
INPUT_FILES = {
    "Rule 1": "seurat_markers_Rule1_MAST_SIGNIFICANT.csv",
    "Rule 2": "seurat_markers_Rule2_MAST_SIGNIFICANT.csv",
    "Rule 4": "seurat_markers_Rule4_MAST_SIGNIFICANT.csv"
}

GENE_SETS = ['GO_Biological_Process_2023']

def run_seurat_validation():
    print("Running Enrichr on Seurat/MAST Results...")

    for rule, filepath in INPUT_FILES.items():
        if not os.path.exists(filepath):
            print(f"Skipping {rule}: File not found.")
            continue
            
        # 1. Load Markers
        df = pd.read_csv(filepath)
        # Handle unnamed column if present
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "Gene"}, inplace=True)
            
        gene_list = df['Gene'].tolist()
        print(f"\n{rule}: Found {len(gene_list)} genes.")
        
        if len(gene_list) < 5:
            print("  Not enough genes.")
            continue
            
        # 2. Run Enrichr
        try:
            enr = gp.enrichr(gene_list=gene_list,
                             gene_sets=GENE_SETS,
                             organism='Human',
                             outdir=None)
            
            results = enr.results
            
            # Filter for significance (p < 0.05)
            sig_results = results[results['Adjusted P-value'] < 0.05].sort_values("Adjusted P-value")
            
            if sig_results.empty:
                print("  No significant pathways found.")
                continue
                
            # 3. Save
            out_name = f"pathway_validation_{rule.replace(' ', '_')}_MAST.csv"
            sig_results.to_csv(out_name, index=False)
            print(f"  Saved {len(sig_results)} pathways to: {out_name}")
            print(sig_results[['Term', 'Adjusted P-value']].head(3))
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    run_seurat_validation()
