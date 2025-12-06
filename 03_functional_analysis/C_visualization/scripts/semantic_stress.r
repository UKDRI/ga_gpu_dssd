# ==============================================================================
# R Script: Violin Plots (Comparing Rules vs. General PD)
# ==============================================================================

suppressPackageStartupMessages({
    library(Seurat)
    library(ggplot2)
    library(dplyr)
})

print("Starting Analysis...")

# 1. Load Data
# ------------------------------------------------------------------------------
load("sc_data/SNatlas_DaNs_seurat.RData")
obj_list <- ls()
seurat_obj_name <- obj_list[obj_list != "obj_list"]
seurat_obj <- get(seurat_obj_name[1])

X <- read.csv("sc_data/X_binary.csv", check.names = FALSE)
cell_ids <- read.csv("sc_data/cell_ids.csv", stringsAsFactors = FALSE)
if (colnames(cell_ids)[1] == "x") { colnames(cell_ids) <- "Barcode" }

# 2. Reconstruct Subgroups
# ------------------------------------------------------------------------------
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

# 3. Label "Control" and "Other PD" (NEW STEP)
# ------------------------------------------------------------------------------
# 1. Label Healthy Controls
control_labels <- c("HC", "Control", "CTR", "Healthy", "HC1")
control_cells <- which(seurat_obj$Disease %in% control_labels)
seurat_obj$Subgroup[control_cells] <- "Control"

# 2. Label "Other PD" (Cells that are PD but didn't hit a Rule)
# We find cells that are NOT Control, but are still "Unassigned" (or NA) in Subgroup
pd_background_cells <- which(
  !seurat_obj$Disease %in% control_labels &
  (is.na(seurat_obj$Subgroup) | seurat_obj$Subgroup == "Unassigned")
)
seurat_obj$Subgroup[pd_background_cells] <- "Other PD"

# Check what groups we have now
print(table(seurat_obj$Subgroup))

# Subset to keep everything (Control, Other PD, Rules)
idents_keep <- c("Control", "Other PD", "Rule 1 (Metabolic)", "Rule 2 (Iron/Stress)", "Rule 4 (Regeneration)")
seurat_subset <- subset(seurat_obj, subset = Subgroup %in% idents_keep)
Idents(seurat_subset) <- "Subgroup"

# 4. Calculate Scores
# ------------------------------------------------------------------------------
stress_genes <- list(c("HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1", "DNAJB1", "HSPE1"))
da_genes <- list(c("TH", "SLC6A3", "SLC18A2", "DDC", "ALDH1A1", "RET", "FOXA2", "LMX1A", "NR4A2"))

seurat_subset <- AddModuleScore(seurat_subset, features = stress_genes, name = "General_Stress")
seurat_subset <- AddModuleScore(seurat_subset, features = da_genes, name = "DA_Identity")

# 5. Plotting Function (Updated Colors & Comparisons)
# ------------------------------------------------------------------------------
# Define Colors (Added Dark Grey for "Other PD")
my_cols <- c("Control" = "lightgrey",
             "Other PD" = "#666666",   # Darker Grey for general disease
             "Rule 1 (Metabolic)" = "#4c72b0",
             "Rule 2 (Iron/Stress)" = "#dd8452",
             "Rule 4 (Regeneration)" = "#55a868")

create_plot_no_deps <- function(data_obj, feature, title, y_lab) {

  plot_data <- data.frame(
    Subgroup = data_obj$Subgroup,
    Score = data_obj@meta.data[[feature]]
  )

  # Set Order: Control -> Other PD -> Rules
  plot_data$Subgroup <- factor(plot_data$Subgroup, levels = c("Control", "Other PD", "Rule 1 (Metabolic)", "Rule 2 (Iron/Stress)", "Rule 4 (Regeneration)"))

  # Calculate Stats
  # We now compare RULES vs OTHER PD (to show they are special)
  groups_to_test <- c("Rule 1 (Metabolic)", "Rule 2 (Iron/Stress)", "Rule 4 (Regeneration)")
  stats_df <- data.frame(Subgroup = character(), Pval = character(), Y_pos = numeric())

  # Use "Other PD" as the baseline for comparison!
  baseline_scores <- plot_data$Score[plot_data$Subgroup == "Other PD"]
  max_val <- max(plot_data$Score, na.rm=TRUE)

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

      # Offset the stars slightly for readability
      stats_df <- rbind(stats_df, data.frame(Subgroup = g, Pval = lbl, Y_pos = max_val * 1.05))
    }
  }

  p <- ggplot(plot_data, aes(x = Subgroup, y = Score, fill = Subgroup)) +
    geom_violin(trim = FALSE, scale = "width", alpha = 0.6, color = NA) +
    geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA, alpha = 0.9) +

    # Add Stars (Comparing Rules vs Other PD)
    geom_text(data = stats_df, aes(x = Subgroup, y = Y_pos, label = Pval),
              size = 6, color = "black", inherit.aes = FALSE) +

    scale_fill_manual(values = my_cols) +
    theme_classic() +
    labs(title = title, y = y_lab, x = "") +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )

  return(p)
}

# Run
p1 <- create_plot_no_deps(seurat_subset, "General_Stress1", "Universal Cellular Stress", "Stress Module Score")
p2 <- create_plot_no_deps(seurat_subset, "DA_Identity1", "Neuronal Identity Integrity", "DA Identity Score")

ggsave("Violin_Stress_vs_OtherPD.png", plot = p1, width = 10, height = 7)
ggsave("Violin_Identity_vs_OtherPD.png", plot = p2, width = 10, height = 7)

print("Analysis Complete. Comparisons are Rules vs 'Other PD'.")
