library(Seurat)
library(ggplot2)
library(dplyr)

# 1. Load Base Object
data_file <- "./sc_data/SNatlas_DaNs_seurat.RData"
print("Loading base Seurat object...")
load(data_file)
seurat_obj <- sn_atlas_dans

# 2. Load Rule Data
print("Loading CSVs...")
X_bin <- read.csv("./sc_data/X_binary.csv")
cell_ids <- read.csv("./sc_data/cell_ids.csv")

# --- DEBUG 1: Check Dimensions ---
print(paste("Binary Matrix Rows:", nrow(X_bin)))
print(paste("Cell ID Rows:", nrow(cell_ids)))

# --- FIX: Robustly get Cell IDs ---
# Just take the first column, whatever it is named (likely 'x')
actual_ids <- cell_ids[,1]

# --- DEBUG 2: Check Gene Columns ---
# Ensure the genes exist in X_bin
req_genes <- c("FTH1", "RBX1", "PALD1", "SLTM", "TMEM219")
missing_genes <- req_genes[!req_genes %in% colnames(X_bin)]
if(length(missing_genes) > 0) {
  print("WARNING: The following genes are missing from X_binary.csv:")
  print(missing_genes)
  # Try replacing hyphens with dots if necessary (common R issue)
  # But FTH1 etc shouldn't have this issue.
}

# 3. Identify Rule 2 Cells
print("Identifying Rule 2 matches...")
# Logic: FTH1=1, others=0
rule2_mask <- (X_bin$FTH1 == 1) &
              (X_bin$RBX1 == 0) &
              (X_bin$PALD1 == 0) &
              (X_bin$SLTM == 0) &
              (X_bin$TMEM219 == 0)

# Count matches
num_matches <- sum(rule2_mask)
print(paste(">>> FOUND", num_matches, "CELLS MATCHING RULE 2 <<<"))

if(num_matches == 0) {
  stop("Stopping: No cells matched the rule. Check your binary matrix values (are they 0/1?).")
}

# Get barcodes
rule2_barcodes <- actual_ids[rule2_mask]

# 4. Filter Seurat Object
# We only want to plot the cells that we actually analyzed (CTR + PD)
# otherwise the grey background will be misleadingly large
seurat_subset <- subset(seurat_obj, cells = actual_ids)


# --- NEW: KEEP ONLY PD CELLS ---
# We assume the metadata column is named 'Disease' or 'ga_label'
# Check your metadata names, but usually it's 'Disease' with "PD_..." values
seurat_subset <- subset(seurat_subset, subset = Disease != "CTR")
# -------------------------------

# 5. Add Metadata
seurat_subset$Rule2_Status <- ifelse(colnames(seurat_subset) %in% rule2_barcodes, "Iron-Toxic", "Other")

# Check if metadata was actually added
print("Status counts in Seurat object:")
print(table(seurat_subset$Rule2_Status))

# 6. PLOT 1: Confetti UMAP
print("Plotting UMAP...")
p1 <- DimPlot(seurat_subset, group.by = "Rule2_Status",
              cols = c("Iron-Toxic" = "#d62728", "Other" = "lightgrey"),
              order = TRUE, pt.size = 0.8) + # Increased dot size slightly
      ggtitle("Rule 2 State (Red) vs Global Clusters") +
      theme(legend.position = "bottom")

ggsave("rule2_umap_confetti.png", plot = p1, width = 6, height = 6)

# 7. PLOT 2: Cluster Breakdown
print("Plotting Cluster Breakdown...")
p2 <- ggplot(seurat_subset@meta.data, aes(x = seurat_clusters, fill = Rule2_Status)) +
      geom_bar(position = "fill") +
      scale_y_continuous(labels = scales::percent) +
      scale_fill_manual(values = c("Iron-Toxic" = "#d62728", "Other" = "lightgrey")) +
      labs(y = "% Cells in Iron State", x = "Seurat Cluster",
           title = "Prevalence of Iron State across Clusters") +
      theme_minimal()

ggsave("rule2_cluster_breakdown.png", plot = p2, width = 8, height = 4)

print("Success. Check 'rule2_umap_confetti.png'.")
