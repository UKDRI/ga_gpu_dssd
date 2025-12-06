import pandas as pd
import gseapy as gp
import os

# --- CONFIGURATION ---

# Map Rule Names to the MAST output files (significant DE genes)
INPUT_FILES = {
    "Rule 1": "seurat_markers_Rule1_MAST_SIGNIFICANT.csv",
    "Rule 2": "seurat_markers_Rule2_MAST_SIGNIFICANT.csv",
    "Rule 4": "seurat_markers_Rule4_MAST_SIGNIFICANT.csv",
}

# Group libraries into "layers" for summarisation
GENE_SET_GROUPS = {
    "go_pathway": [
        # GO + pathways
        "GO_Biological_Process_2025",
        "GO_Cellular_Component_2025",
        "GO_Molecular_Function_2025",
        "Reactome_2022",
        "KEGG_2021_Human",
        "WikiPathway_2023_Human",
    ],
    "neuron_synapse": [
        # Neuron / synapse context
        "SynGO_2024",
    ],
    "disease_genetics": [
        # Disease / genetics
        "Jensen_DISEASES",
        "Jensen_DISEASES_Curated_2025",
        "GWAS_Catalog_2023",
        "Human_Phenotype_Ontology",
        "OMIM_Disease",
    ],
    "regulators": [
        # TFs / miRNAs
        "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
        "ChEA_2022",
        "TRRUST_Transcription_Factors_2019",
        "TRANSFAC_and_JASPAR_PWMs",
        "TargetScan_microRNA_2017",
    ],
}

# Flattened list of all gene sets to query
GENE_SETS = [lib for libs in GENE_SET_GROUPS.values() for lib in libs]

# Custom background: all genes tested in your MAST model
BACKGROUND_FILE = "mast_background_all_genes.txt"


def run_seurat_validation():
    print("Running Enrichr on Seurat/MAST Results with custom background...")

    # --- Load background once ---
    if not os.path.exists(BACKGROUND_FILE):
        raise FileNotFoundError(
            f"Background file '{BACKGROUND_FILE}' not found. "
            "Export all MAST-tested genes (one per line) from Seurat and save to this path."
        )

    bg_df = pd.read_csv(BACKGROUND_FILE, header=None)
    background_genes = bg_df.iloc[:, 0].astype(str).str.strip().tolist()
    print(f"Loaded {len(background_genes)} background genes from {BACKGROUND_FILE}")

    for rule, filepath in INPUT_FILES.items():
        if not os.path.exists(filepath):
            print(f"Skipping {rule}: File not found ({filepath}).")
            continue

        # 1. Load significant markers for this rule
        df = pd.read_csv(filepath)

        # Handle unnamed column if present
        if "Unnamed: 0" in df.columns and "Gene" not in df.columns:
            df.rename(columns={"Unnamed: 0": "Gene"}, inplace=True)

        if "Gene" not in df.columns:
            print(f"{rule}: No 'Gene' column found in {filepath}, skipping.")
            continue

        gene_list = df["Gene"].astype(str).str.strip().tolist()
        print(f"\n{rule}: Found {len(gene_list)} significant genes before filtering by background.")

        # Restrict gene list to those in background (defensive)
        gene_list = [g for g in gene_list if g in background_genes]
        print(f"{rule}: {len(gene_list)} genes after intersecting with background.")

        if len(gene_list) < 5:
            print("  Not enough genes after background filtering.")
            continue

        all_results = []

        # 2. Run Enrichr PER LIBRARY, catching encoding errors
        print(f"  Running Enrichr for {len(GENE_SETS)} libraries...")
        for lib in GENE_SETS:
            try:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=[lib],
                    background=background_genes,   # can also be BACKGROUND_FILE as a path
                    outdir=None,                   # don't write plots/logs
                    verbose=False,
                )
                res = enr.results
                if res is None or res.empty:
                    print(f"    {lib}: no results.")
                    continue

                # Some versions already include a column naming the library; keep as-is.
                # Just ensure there's a 'Gene_set' column for later grouping.
                if "Gene_set" not in res.columns:
                    res["Gene_set"] = lib

                all_results.append(res)
                print(f"    {lib}: OK, {len(res)} rows.")

            except Exception as e:
                # This will catch UnicodeDecodeError and similar
                print(f"    {lib}: SKIPPED due to error -> {e}")

        if not all_results:
            print("  No libraries returned usable results for this rule.")
            continue

        # 3. Concatenate results from all successful libraries
        results = pd.concat(all_results, ignore_index=True)

        if "Adjusted P-value" not in results.columns:
            print("  'Adjusted P-value' column not found in results, skipping.")
            continue

        # 4. Filter for significance (FDR < 0.05)
        sig_results = (
            results[results["Adjusted P-value"] < 0.05]
            .sort_values("Adjusted P-value")
        )

        if sig_results.empty:
            print("  No significant pathways found (FDR < 0.05).")
            continue

        # 5. Save combined significant results for this rule
        base_name = f"pathway_validation_{rule.replace(' ', '_')}_MAST_bg"
        combined_out = f"{base_name}.csv"
        sig_results.to_csv(combined_out, index=False)
        print(f"  Saved {len(sig_results)} significant terms to: {combined_out}")
        print(sig_results[["Gene_set", "Term", "Adjusted P-value"]].head(5))

        # 6. Per-layer summaries
        gene_set_col = None
        for cand in ["Gene_set", "Gene_set_library", "Library"]:
            if cand in sig_results.columns:
                gene_set_col = cand
                break

        if gene_set_col is None:
            print("  Could not find a gene-set column to split by layer; skipping per-layer files.")
            continue

        for layer_name, libs in GENE_SET_GROUPS.items():
            layer_df = sig_results[sig_results[gene_set_col].isin(libs)]
            if layer_df.empty:
                print(f"  {layer_name}: no significant terms.")
                continue

            layer_out = f"{base_name}_{layer_name}.csv"
            layer_df.to_csv(layer_out, index=False)
            print(f"  {layer_name}: saved {len(layer_df)} terms to {layer_out}")
            print(layer_df[["Gene_set", "Term", "Adjusted P-value"]].head(3))


if __name__ == "__main__":
    run_seurat_validation()
