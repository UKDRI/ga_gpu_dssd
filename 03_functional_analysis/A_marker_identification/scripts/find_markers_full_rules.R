library(Seurat)
library(dplyr)

# --- CONFIGURATION ---
print("Loading Seurat Object...")
load("./sc_data/SNatlas_DaNs_seurat.RData")
seurat_obj <- sn_atlas_dans

print("Loading Rule Definitions...")
X_bin <- read.csv("./sc_data/X_binary.csv", check.names = FALSE)
cell_ids_file <- read.csv("./sc_data/cell_ids.csv")

# Robust ID matching
cell_barcodes <- cell_ids_file[,1]
if (length(cell_barcodes) != nrow(X_bin)) {
  stop("Error: cell_ids row count mismatch!")
}
rownames(X_bin) <- cell_barcodes

# --- HELPER: Get Cells ---
get_rule_cells <- function(data, conditions) {
  mask <- rep(TRUE, nrow(data))
  for (cond in conditions) {
    gene <- cond$gene
    val <- cond$val
    if (!gene %in% colnames(data)) next
    mask <- mask & (data[[gene]] == val)
  }
  return(rownames(data)[mask])
}

# --- DEFINE RULES (Full Strict Versions) ---
rule1_conds <- list(
  list(gene="NCOR2", val=0), list(gene="FAM19A5", val=0), list(gene="BRSK2", val=0),
  list(gene="MT-ND4L", val=1), list(gene="BIN1", val=0), list(gene="SEMA4D", val=0),
  list(gene="FAM102A", val=0), list(gene="ZFYVE16", val=0), list(gene="ST3GAL4", val=0),
  list(gene="ATP6V1D", val=0), list(gene="DSTN", val=0), list(gene="CHST3", val=0),
  list(gene="PALD1", val=0), list(gene="COX4I1", val=0), list(gene="FNTA", val=0)
)
rule2_conds <- list(
  list(gene="FTH1", val=1), list(gene="RBX1", val=0), list(gene="PALD1", val=0),
  list(gene="SLTM", val=0), list(gene="TMEM219", val=0)
)
rule4_conds <- list(list(gene="PALD1", val=1))

# --- ANALYSIS FUNCTION ---
run_de_analysis <- function(seurat_obj, rule_name, conditions, output_prefix) {
  print(paste0("\n=== Analyzing ", rule_name, " ==="))

  target_barcodes <- get_rule_cells(X_bin, conditions)

  # Filter: PD Only
  pd_cells <- colnames(seurat_obj)[seurat_obj$Disease != "CTR"]
  target_pd_cells <- intersect(target_barcodes, pd_cells)
  background_pd_cells <- setdiff(pd_cells, target_pd_cells)

  print(paste("  Target PD Cells:", length(target_pd_cells)))

  if(length(target_pd_cells) < 3) {
    print("  Skipping: Too few target cells.")
    return(NULL)
  }

  # Set Identity
  cells_to_use <- c(target_pd_cells, background_pd_cells)
  seurat_sub <- subset(seurat_obj, cells = cells_to_use)
  seurat_sub$DE_Group <- ifelse(colnames(seurat_sub) %in% target_pd_cells, "Subgroup", "Background")
  Idents(seurat_sub) <- "DE_Group"

  # Run FindMarkers
  print("  Running FindMarkers (Wilcoxon)...")
  markers <- FindMarkers(seurat_sub,
                         ident.1 = "Subgroup",
                         ident.2 = "Background",
                         test.use = "wilcox",
                         logfc.threshold = 0.25,
                         min.pct = 0.1)

  # --- FILTERING LOGIC ---
  # 1. Add difference column
  markers$diff_pct <- markers$pct.1 - markers$pct.2

  # 2. Define Significance Criteria (Strict)
  # p < 0.01, LogFC > 0.5, Diff > 15%
  sig_mask <- (markers$p_val_adj < 0.01) &
              (markers$avg_log2FC > 0.5)# &
              #(markers$diff_pct > 0.15)

  sig_genes <- markers[sig_mask, ]

  # Sort by P-value
  sig_genes <- sig_genes[order(sig_genes$p_val_adj, decreasing = FALSE), ]

  # --- SAVE FILES ---
  # 1. Save ALL results (for reference)
  write.csv(markers, paste0("seurat_markers_", output_prefix, "_ALL.csv"))

  # 2. Save SIGNIFICANT results (for paper)
  outfile_sig = paste0("seurat_markers_", output_prefix, "_SIGNIFICANT.csv")
  write.csv(sig_genes, outfile_sig)

  print(paste("  Saved significant markers to:", outfile_sig))
  print(paste("  Total Significant Markers found:", nrow(sig_genes)))
  print("  Top 5 Significant Hits:")
  print(head(sig_genes, 5))
}

# --- EXECUTE ---
run_de_analysis(seurat_obj, "Rule 1 (Metabolic)", rule1_conds, "Rule1")
run_de_analysis(seurat_obj, "Rule 2 (Iron)", rule2_conds, "Rule2")
run_de_analysis(seurat_obj, "Rule 4 (Fibrotic)", rule4_conds, "Rule4")

print("\nAll DE analyses complete.")
