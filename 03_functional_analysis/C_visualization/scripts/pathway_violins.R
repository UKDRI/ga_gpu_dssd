# ==============================================================================
# R Script: Batch Functional Scoring (Violin Plots from File Lists)
# ==============================================================================

suppressPackageStartupMessages({
    library(Seurat)
    library(ggplot2)
    library(dplyr)
})

print("Starting Analysis...")

# ------------------------------------------------------------------------------
# 1. Load Data & Reconstruct Subgroups (Standard Setup)
# ------------------------------------------------------------------------------
# Load Seurat
if(file.exists("sc_data/SNatlas_DaNs_seurat.RData")){
  load("sc_data/SNatlas_DaNs_seurat.RData")
} else {
  stop("Seurat RData file not found.")
}

obj_list <- ls()
seurat_obj_name <- obj_list[obj_list != "obj_list"]
seurat_obj <- get(seurat_obj_name[1])

# Load Rules and IDs
X <- read.csv("sc_data/X_binary.csv", check.names = FALSE)
cell_ids <- read.csv("sc_data/cell_ids.csv", stringsAsFactors = FALSE)
if (colnames(cell_ids)[1] == "x") { colnames(cell_ids) <- "Barcode" }

X$Barcode <- cell_ids$Barcode
X$Subgroup <- "Unassigned"

# Apply Rules
r1_mask <- with(X, `MT-ND4L` == 1 & NCOR2 == 0 & FAM19A5 == 0 & BRSK2 == 0 &
                  BIN1 == 0 & SEMA4D == 0 & FAM102A == 0 & ZFYVE16 == 0 &
                  ST3GAL4 == 0 & ATP6V1D == 0 & DSTN == 0 & CHST3 == 0 &
                  PALD1 == 0 & COX4I1 == 0 & FNTA == 0)
X$Subgroup[r1_mask] <- "Rule 1 (Metabolic)"

r2_mask <- with(X, FTH1 == 1 & RBX1 == 0 & PALD1 == 0 & SLTM == 0 & TMEM219 == 0)
X$Subgroup[r2_mask] <- "Rule 2 (Iron/Stress)"

r4_mask <- with(X, PALD1 == 1)
X$Subgroup[r4_mask] <- "Rule 4 (Regeneration)"

# Add to Seurat
meta_rules <- X[, c("Barcode", "Subgroup")]
rownames(meta_rules) <- meta_rules$Barcode
seurat_obj <- AddMetaData(seurat_obj, metadata = meta_rules$Subgroup, col.name = "Subgroup")

# Label "Control" and "Other PD"
control_labels <- c("HC", "Control", "CTR", "Healthy", "HC1")
control_cells <- which(seurat_obj$Disease %in% control_labels)
seurat_obj$Subgroup[control_cells] <- "Control"

pd_background_cells <- which(
  !seurat_obj$Disease %in% control_labels &
  (is.na(seurat_obj$Subgroup) | seurat_obj$Subgroup == "Unassigned")
)
seurat_obj$Subgroup[pd_background_cells] <- "Other PD"

# Subset Data
idents_keep <- c("Control", "Other PD", "Rule 1 (Metabolic)", "Rule 2 (Iron/Stress)", "Rule 4 (Regeneration)")
seurat_subset <- subset(seurat_obj, subset = Subgroup %in% idents_keep)
Idents(seurat_subset) <- "Subgroup"

# ------------------------------------------------------------------------------
# 2. Configuration: Define your Gene Lists
# ------------------------------------------------------------------------------
# Format: "Filename" = "Plot Title"
# Ensure these files exist in the 'gene_lists' directory
files_map <- list(
  "raas_genes.txt"                 = "RAAS Pathway",
  "mapk_genes_in_wikipathways.txt" = "MAPK Signaling (WikiPathways)",
  "dan3_regulon_TCF7L2_targets"    = "TCF7L2 Targets (Dan3 Regulon)", # Assuming no extension based on your prompt
  "DopamineKEGGpathway.txt"        = "Dopamine Metabolism (KEGG)"
)

# ------------------------------------------------------------------------------
# 3. Plotting Function (No Dependencies)
# ------------------------------------------------------------------------------
my_cols <- c("Control" = "lightgrey",
             "Other PD" = "#666666",
             "Rule 1 (Metabolic)" = "#4c72b0",
             "Rule 2 (Iron/Stress)" = "#dd8452",
             "Rule 4 (Regeneration)" = "#55a868")

create_plot_no_deps <- function(data_obj, feature, title, y_lab) {

  plot_data <- data.frame(
    Subgroup = data_obj$Subgroup,
    Score = data_obj@meta.data[[feature]]
  )
  plot_data$Subgroup <- factor(plot_data$Subgroup, levels = c("Control", "Other PD", "Rule 1 (Metabolic)", "Rule 2 (Iron/Stress)", "Rule 4 (Regeneration)"))

  # Stats: Compare Rules vs Other PD
  groups_to_test <- c("Rule 1 (Metabolic)", "Rule 2 (Iron/Stress)", "Rule 4 (Regeneration)")
  stats_df <- data.frame(Subgroup = character(), Pval = character(), Y_pos = numeric())

  baseline_scores <- plot_data$Score[plot_data$Subgroup == "Other PD"]
  max_val <- max(plot_data$Score, na.rm=TRUE)
  min_val <- min(plot_data$Score, na.rm=TRUE)

  # Calculate Wilcoxon P-values
  for(g in groups_to_test) {
    g_scores <- plot_data$Score[plot_data$Subgroup == g]
    if(length(g_scores) > 0) {
      res <- wilcox.test(g_scores, baseline_scores)
      p <- res$p.value

      if(p < 0.0001) { lbl <- "****" }
      else if(p < 0.001) { lbl <- "***" }
      else if(p < 0.01) { lbl <- "**" }
      else if(p < 0.05) { lbl <- "*" }
      else { lbl <- "ns" }

      stats_df <- rbind(stats_df, data.frame(Subgroup = g, Pval = lbl, Y_pos = max_val + (max_val - min_val)*0.05))
    }
  }

  p <- ggplot(plot_data, aes(x = Subgroup, y = Score, fill = Subgroup)) +
    geom_violin(trim = FALSE, scale = "width", alpha = 0.6, color = NA) +
    geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA, alpha = 0.9) +
    geom_text(data = stats_df, aes(x = Subgroup, y = Y_pos, label = Pval),
              size = 6, color = "black", inherit.aes = FALSE) +
    scale_fill_manual(values = my_cols) +
    theme_classic() +
    labs(title = title, y = y_lab, x = "") +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    )
  return(p)
}

# ------------------------------------------------------------------------------
# 4. Processing Loop
# ------------------------------------------------------------------------------
print("Processing gene lists...")

for (filename in names(files_map)) {

  plot_title <- files_map[[filename]]
  file_path <- file.path("gene_lists", filename)

  print(paste("Reading:", file_path))

  if (file.exists(file_path)) {
    # Read genes (assuming one gene per line or separated by whitespace)
    genes <- scan(file_path, what = "", quiet = TRUE)

    # Check if genes exist in Seurat object
    valid_genes <- intersect(genes, rownames(seurat_subset))

    if (length(valid_genes) > 0) {
      print(paste("  Found", length(valid_genes), "valid genes."))

      # Define a safe column name (remove spaces/special chars)
      safe_name <- gsub("[^A-Za-z0-9]", "_", filename)

      # Calculate Score
      # AddModuleScore appends '1' to the name provided
      seurat_subset <- AddModuleScore(seurat_subset, features = list(valid_genes), name = safe_name)

      # The actual column name created by Seurat
      col_name <- paste0(safe_name, "1")

      # Generate Plot
      p <- create_plot_no_deps(seurat_subset, col_name, plot_title, "Module Score")

      # Save Plot
      out_name <- paste0("Violin_", safe_name, ".png")
      ggsave(out_name, plot = p, width = 10, height = 7)
      print(paste("  Saved:", out_name))

    } else {
      print("  WARNING: None of the genes in this file were found in the dataset.")
    }
  } else {
    print(paste("  ERROR: File not found:", file_path))
  }
}

print("Batch Analysis Complete!")
