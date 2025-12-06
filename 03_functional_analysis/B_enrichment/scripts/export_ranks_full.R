library(Seurat)
library(dplyr)

# --- CONFIGURATION ---
print("Loading Data...")
load("./sc_data/SNatlas_DaNs_seurat.RData")
seurat_obj <- sn_atlas_dans
X_bin <- read.csv("./sc_data/X_binary.csv", check.names = FALSE)
cell_ids <- read.csv("./sc_data/cell_ids.csv")
rownames(X_bin) <- cell_ids[,1]

# --- HELPER: Identify Cells ---
get_rule_cells <- function(data, conditions) {
  mask <- rep(TRUE, nrow(data))
  for (cond in conditions) {
    if (!cond$gene %in% colnames(data)) next
    mask <- mask & (data[[cond$gene]] == cond$val)
  }
  return(rownames(data)[mask])
}

# --- THE FULL STRICT RULES (15 genes for Rule 1) ---
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

rule4_conds <- list(
  list(gene="PALD1", val=1)
)

# --- EXPORT FUNCTION ---
export_rank_list <- function(seurat_obj, rule_name, conditions, output_prefix) {
  print(paste0("\n=== Generating Ranks for ", rule_name, " ==="))

  # 1. Identify Cells (Strict Definition)
  target_barcodes <- get_rule_cells(X_bin, conditions)

  # 2. Filter PD Only
  pd_cells <- colnames(seurat_obj)[seurat_obj$Disease != "CTR"]
  target_pd <- intersect(target_barcodes, pd_cells)
  bg_pd <- setdiff(pd_cells, target_pd)

  print(paste("  Target Cells:", length(target_pd)))

  if(length(target_pd) < 3) {
    print("Skipping: Too few cells.")
    return(NULL)
  }

  # 3. Subset
  sub_obj <- subset(seurat_obj, cells = c(target_pd, bg_pd))
  sub_obj$Group <- ifelse(colnames(sub_obj) %in% target_pd, "Target", "Bg")
  Idents(sub_obj) <- "Group"

  # 4. Run Wilcoxon on ALL genes (logfc.threshold=0)
  # We need every gene to create a full ranked list for GSEA
  print("  Running Differential Expression on all genes...")
  markers <- FindMarkers(sub_obj, ident.1 = "Target", ident.2 = "Bg",
                         test.use = "wilcox",
                         logfc.threshold = 0,
                         min.pct = 0.01, # Allow rare genes if they are specific
                         verbose = FALSE)

  # 5. Create Rank Metric
  # Metric = -log10(p_value) * sign(log_fold_change)
  # We add 1e-300 to p-value to avoid log(0) = Infinity
  markers$rank_metric <- -log10(markers$p_val + 1e-300) * sign(markers$avg_log2FC)

  # 6. Save .rnk file
  rank_df <- data.frame(Gene = rownames(markers), Rank = markers$rank_metric)
  # Sort descending (High Positive at top, Low Negative at bottom)
  rank_df <- rank_df[order(rank_df$Rank, decreasing = TRUE), ]

  outfile <- paste0(output_prefix, "_FULL.rnk")
  write.table(rank_df, outfile, sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)
  print(paste("  Saved:", outfile))
}

# --- EXECUTE ---
export_rank_list(seurat_obj, "Rule 1 (Strict)", rule1_conds, "Rule1")
export_rank_list(seurat_obj, "Rule 2 (Strict)", rule2_conds, "Rule2")
export_rank_list(seurat_obj, "Rule 4 (Strict)", rule4_conds, "Rule4")
