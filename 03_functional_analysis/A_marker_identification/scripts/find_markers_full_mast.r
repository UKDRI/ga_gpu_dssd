library(Seurat)
library(dplyr)

# --- CONFIGURATION ---
print("Loading Data...")
load("./sc_data/SNatlas_DaNs_seurat.RData")
seurat_obj <- sn_atlas_dans
X_bin <- read.csv("./sc_data/X_binary.csv", check.names = FALSE)
cell_ids <- read.csv("./sc_data/cell_ids.csv")
rownames(X_bin) <- cell_ids[,1]

# --- HELPER ---
get_rule_cells <- function(data, conditions) {
  mask <- rep(TRUE, nrow(data))
  for (cond in conditions) {
    if (!cond$gene %in% colnames(data)) next
    mask <- mask & (data[[cond$gene]] == cond$val)
  }
  return(rownames(data)[mask])
}

# --- RULES (Strict) ---
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

# --- ANALYSIS ---
run_mast_analysis <- function(seurat_obj, rule_name, conditions, output_prefix) {
  print(paste0("\n=== Analyzing ", rule_name, " (MAST) ==="))

  target_barcodes <- get_rule_cells(X_bin, conditions)

  # Filter PD Only
  pd_cells <- colnames(seurat_obj)[seurat_obj$Disease != "CTR"]
  target_pd <- intersect(target_barcodes, pd_cells)
  bg_pd <- setdiff(pd_cells, target_pd)

  if(length(target_pd) < 3) return(NULL)

  # Subset
  sub_obj <- subset(seurat_obj, cells = c(target_pd, bg_pd))
  sub_obj$Group <- ifelse(colnames(sub_obj) %in% target_pd, "Target", "Bg")
  Idents(sub_obj) <- "Group"

                                        # --- THE CHANGE: test.use = "MAST" ---


  print("  Running FindMarkers (MAST)... this may take a moment.")
  # Note: MAST works best on LogNormalized data, but Seurat handles SCT automatically.
  # We keep logfc.threshold=0 to capture everything, then filter later.
  markers <- FindMarkers(sub_obj,
                         ident.1 = "Target",
                         ident.2 = "Bg",
                         test.use = "MAST",
                         min.pct = 0.1,
                         logfc.threshold = 0.25)

  # Filter & Sort (Same Criteria as before)
  # p < 0.01, logFC > 0.25
  sig_genes <- subset(markers, p_val_adj < 0.01 & avg_log2FC > 0.25)
  sig_genes <- sig_genes[order(sig_genes$p_val_adj), ]

  outfile <- paste0("seurat_markers_", output_prefix, "_MAST_SIGNIFICANT.csv")
  write.csv(sig_genes, outfile)
  print(paste("  Saved:", outfile))
  print(head(sig_genes, 5))
}

# --- EXECUTE ---
run_mast_analysis(seurat_obj, "Rule 1", rule1_conds, "Rule1")
run_mast_analysis(seurat_obj, "Rule 2", rule2_conds, "Rule2")
run_mast_analysis(seurat_obj, "Rule 4", rule4_conds, "Rule4")


all_background_genes <- rownames(seurat_obj)
writeLines(all_background_genes, "mast_background_all_genes.txt")
print("Saved mast_background_all_genes.txt with full gene universe.")
