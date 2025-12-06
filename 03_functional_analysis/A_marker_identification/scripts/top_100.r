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

# --- RULES ---
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
run_mast_top100 <- function(seurat_obj, rule_name, conditions, output_prefix) {
  print(paste0("\n=== Analyzing ", rule_name, " (MAST Top 100) ==="))

  target_barcodes <- get_rule_cells(X_bin, conditions)
  pd_cells <- colnames(seurat_obj)[seurat_obj$Disease != "CTR"]
  target_pd <- intersect(target_barcodes, pd_cells)
  bg_pd <- setdiff(pd_cells, target_pd)

  if(length(target_pd) < 3) return(NULL)

  sub_obj <- subset(seurat_obj, cells = c(target_pd, bg_pd))
  sub_obj$Group <- ifelse(colnames(sub_obj) %in% target_pd, "Target", "Bg")
  Idents(sub_obj) <- "Group"

  print("  Running FindMarkers...")
  # Loose thresholds to get enough genes
  markers <- FindMarkers(sub_obj, ident.1 = "Target", ident.2 = "Bg",
                         test.use = "MAST", min.pct = 0.1, logfc.threshold = 0)

  # Filter: Positive Enrichment Only
  up_genes <- markers[markers$avg_log2FC > 0, ]

  # Sort by P-value
  up_genes <- up_genes[order(up_genes$p_val_adj), ]

  # Take Top 100
  top_100 <- head(up_genes, 100)

  outfile <- paste0("seurat_markers_", output_prefix, "_MAST_TOP100.csv")
  write.csv(top_100, outfile)
  print(paste("  Saved Top 100 to:", outfile))
}

# --- EXECUTE ---
run_mast_top100(seurat_obj, "Rule 1", rule1_conds, "Rule1")
run_mast_top100(seurat_obj, "Rule 2", rule2_conds, "Rule2")
run_mast_top100(seurat_obj, "Rule 4", rule4_conds, "Rule4")
