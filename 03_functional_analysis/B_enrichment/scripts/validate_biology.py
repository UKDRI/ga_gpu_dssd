import pandas as pd
import glob
import os
import gseapy as gp
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# The databases to query. 
# 'GO_Biological_Process_2023' gives you processes (e.g., "Mitochondrial Assembly")
# 'Jensen_DISEASES' gives you disease associations (e.g., "Parkinson's Disease")
GENE_SETS = ['GO_Biological_Process_2023', 'Jensen_DISEASES']

def run_enrichment(csv_file):
    print(f"\nProcessing: {csv_file}...")
    
    # 1. Load the differential enrichment results
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        print("  Skipping: Could not read file.")
        return

    # 2. Filter for UP-regulated genes only
    # We want to know what this subgroup IS doing, not what it stopped doing.
    # We take genes where the Difference > 0.05 (5%)
    up_genes = df[df['Difference'] > 0.05]
    
    # 3. Get the gene list (Top 100 max)
    gene_list = up_genes['Gene'].tolist()[:100]
    
    # Clean gene names (sometimes excel/csv adds spaces)
    gene_list = [str(g).strip() for g in gene_list]
    
    print(f"  Found {len(gene_list)} enriched genes to query.")
    
    if len(gene_list) < 5:
        print("  Not enough genes for enrichment analysis (<5).")
        return

    # 4. Run the Enrichment via API (requires internet)
    try:
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=GENE_SETS,
                         organism='Human', 
                         outdir=None) # Don't save raw files yet
        
        results = enr.results
        
        # Filter for significant results (Adjusted P-value < 0.05)
        sig_results = results[results['Adjusted P-value'] < 0.05].copy()
        
        if sig_results.empty:
            print("  No statistically significant pathways found.")
            return

        # 5. Print the Top 5 "Truths"
        # Sort by significance
        sig_results = sig_results.sort_values("Adjusted P-value")
        
        print(f"  --- CONFIRMED BIOLOGY ({len(sig_results)} pathways found) ---")
        top_5 = sig_results[['Gene_set', 'Term', 'Adjusted P-value', 'Overlap']].head(5)
        print(top_5.to_string(index=False))
        
        # 6. Save to CSV for your paper
        out_name = csv_file.replace("enrichment_", "pathway_validation_")
        sig_results.to_csv(out_name, index=False)
        print(f"  Saved full validation to: {out_name}")
        
    except Exception as e:
        print(f"  Error running enrichment: {e}")

# --- MAIN ---
if __name__ == "__main__":
    # Find all the enrichment CSVs created by the previous step
    files = glob.glob("enrichment_*.csv")
    
    if not files:
        print("No 'enrichment_*.csv' files found. Did you run the previous script?")
    else:
        print(f"Found {len(files)} files to validate.")
        for f in files:
            run_enrichment(f)
