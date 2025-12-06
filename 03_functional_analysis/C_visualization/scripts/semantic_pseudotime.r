# ==============================================================================
# R Script: Semantic Time vs. Pseudotime (Robust Fix)
# ==============================================================================

suppressPackageStartupMessages({
    library(Seurat)
    library(ggplot2)
    library(dplyr)
})

print("Starting Analysis...")

# ------------------------------------------------------------------------------
# 1. Load Seurat Object
# ------------------------------------------------------------------------------
print("Loading Seurat Object...")
load("sc_data/SNatlas_DaNs_seurat.RData")

# Handle variable name automatically
obj_list <- ls()
seurat_obj_name <- obj_list[obj_list != "obj_list"]
seurat_obj <- get(seurat_obj_name[1])

# Store the Seurat Barcodes for matching
seurat_barcodes <- Cells(seurat_obj)
print(paste("Seurat object loaded with", length(seurat_barcodes), "cells."))
print(paste("Sample Seurat Barcode:", seurat_barcodes[1]))

# ------------------------------------------------------------------------------
# 2. Load PRE-CALCULATED Pseudotime (The "Matchmaker" Step)
# ------------------------------------------------------------------------------
print("Loading Pseudotime from File...")
traj_data <- read.csv("sc_data/pd_metadata_with_pseudotime.csv", stringsAsFactors = FALSE)

# --- FIX: Find the Barcode Column Automatically ---
# We check every column to see which one overlaps with Seurat barcodes
found_match <- FALSE
barcode_col <- NULL

for (col in colnames(traj_data)) {
  # Check overlap
  overlap <- length(intersect(traj_data[[col]], seurat_barcodes))

  if (overlap > 0) {
    print(paste("Found matching barcode column:", col, "with", overlap, "overlaps."))
    barcode_col <- col
    found_match <- TRUE
    break
  }
}

# If no direct match, try replacing '.' with '-' (Common R CSV issue)
if (!found_match) {
  print("No direct match found. Trying to fix formatted barcodes (converting '.' to '-')...")
  for (col in colnames(traj_data)) {
    # Create temp version with dots replaced
    temp_vals <- gsub("\\.", "-", traj_data[[col]])
    overlap <- length(intersect(temp_vals, seurat_barcodes))

    if (overlap > 0) {
      print(paste("Found match in column", col, "after fixing formatting."))
      traj_data[[col]] <- temp_vals # Update the data with fixed barcodes
      barcode_col <- col
      found_match <- TRUE
      break
    }
  }
}

if (!found_match) {
  print("ERROR: Could not match CSV barcodes to Seurat object.")
  print("Seurat Barcodes look like:")
  print(head(seurat_barcodes))
  print("CSV Columns look like:")
  print(head(traj_data))
  stop("Execution halted due to barcode mismatch.")
}

# ------------------------------------------------------------------------------
# 3. Add Pseudotime to Seurat
# ------------------------------------------------------------------------------
# Identify the time column
time_col <- "pseudotime"
if (!time_col %in% colnames(traj_data)) {
    if ("pseudotime_proxy" %in% colnames(traj_data)) {
        time_col <- "pseudotime_proxy"
    } else {
        stop("Could not find 'pseudotime' or 'pseudotime_proxy' column in CSV!")
    }
}
print(paste("Using time column:", time_col))

# Create the vector
time_vec <- traj_data[[time_col]]
names(time_vec) <- traj_data[[barcode_col]] # Use the identified barcode column

# Add to Seurat (This should work now)
seurat_obj <- AddMetaData(seurat_obj, metadata = time_vec, col.name = "Global_Pseudotime")

# ------------------------------------------------------------------------------
# 4. Load & Add Subgroups
# ------------------------------------------------------------------------------
print("Loading Subgroup Definitions...")
X <- read.csv("sc_data/X_binary.csv", check.names = FALSE)
cell_ids <- read.csv("sc_data/cell_ids.csv", stringsAsFactors = FALSE)

# Fix Cell IDs header
if (colnames(cell_ids)[1] == "x") {
  colnames(cell_ids) <- "Barcode"
}

# Reconstruct Subgroups
X$Barcode <- cell_ids$Barcode
X$Subgroup <- "Unassigned"

# Rule 1: Metabolic
r1_mask <- with(X, `MT-ND4L` == 1 & NCOR2 == 0 & FAM19A5 == 0 & BRSK2 == 0 &
                  BIN1 == 0 & SEMA4D == 0 & FAM102A == 0 & ZFYVE16 == 0 &
                  ST3GAL4 == 0 & ATP6V1D == 0 & DSTN == 0 & CHST3 == 0 &
                  PALD1 == 0 & COX4I1 == 0 & FNTA == 0)
X$Subgroup[r1_mask] <- "Rule 1 (Metabolic)"

# Rule 2: Iron/Stress
r2_mask <- with(X, FTH1 == 1 & RBX1 == 0 & PALD1 == 0 & SLTM == 0 & TMEM219 == 0)
X$Subgroup[r2_mask] <- "Rule 2 (Iron/Stress)"

# Rule 4: Regeneration
r4_mask <- with(X, PALD1 == 1)
X$Subgroup[r4_mask] <- "Rule 4 (Regeneration)"

# Add to Seurat
meta_rules <- X[, c("Barcode", "Subgroup")]
rownames(meta_rules) <- meta_rules$Barcode
seurat_obj <- AddMetaData(seurat_obj, metadata = meta_rules$Subgroup, col.name = "Subgroup")

# Label Controls
control_labels <- c("HC", "Control", "CTR", "Healthy", "HC1")
control_cells <- which(seurat_obj$Disease %in% control_labels)
seurat_obj$Subgroup[control_cells] <- "Control"

# ------------------------------------------------------------------------------
# 5. Subset Data
# ------------------------------------------------------------------------------
print("Subsetting Data...")
idents_keep <- c("Control", "Rule 1 (Metabolic)", "Rule 2 (Iron/Stress)", "Rule 4 (Regeneration)")
seurat_subset <- subset(seurat_obj, subset = Subgroup %in% idents_keep)
Idents(seurat_subset) <- "Subgroup"

# Filter out cells with missing pseudotime (NA)
seurat_subset <- subset(seurat_subset, subset = !is.na(Global_Pseudotime))
print(paste("Cells remaining after filtering:", ncol(seurat_subset)))

# ------------------------------------------------------------------------------
# 6. Calculate Semantic Scores (Identity & Stress)
# ------------------------------------------------------------------------------
print("Calculating Scores...")

# Clock A: "DA Identity"
da_genes <- list(c(
  "TH", "SLC6A3", "SLC18A2", "DDC", "ALDH1A1",
  "RET", "FOXA2", "LMX1A", "NR4A2",
  "SYT1", "SNAP25", "VAMP2", "STX1A"
))

# Clock B: "General Stress"
stress_genes <- list(c(
  "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1", "DNAJB1", "HSPE1"
))

seurat_subset <- AddModuleScore(seurat_subset, features = da_genes, name = "DA_Score")
seurat_subset <- AddModuleScore(seurat_subset, features = stress_genes, name = "General_Stress")

# ------------------------------------------------------------------------------
# 7. Plotting
# ------------------------------------------------------------------------------
print("Generating Plots...")

# Auto-flip Pseudotime?
mean_ctrl <- mean(seurat_subset$Global_Pseudotime[seurat_subset$Subgroup == "Control"], na.rm=TRUE)
mean_r4   <- mean(seurat_subset$Global_Pseudotime[seurat_subset$Subgroup == "Rule 4 (Regeneration)"], na.rm=TRUE)

if (!is.na(mean_ctrl) && !is.na(mean_r4) && mean_ctrl > mean_r4) {
  print("Flipping Pseudotime Axis (Control was > Disease)...")
  seurat_subset$Global_Pseudotime <- seurat_subset$Global_Pseudotime * -1
}

# Create Plot Data
plot_data <- data.frame(
  Global_Time = seurat_subset$Global_Pseudotime,
  DA_Identity = seurat_subset$DA_Score1,
  General_Stress = seurat_subset$General_Stress1,
  Subgroup = seurat_subset$Subgroup
)

# Colors
my_cols <- c("Control" = "lightgrey",
             "Rule 1 (Metabolic)" = "#4c72b0",
             "Rule 2 (Iron/Stress)" = "#dd8452",
             "Rule 4 (Regeneration)" = "#55a868")

# PLOT: Identity Loss
p1 <- ggplot(plot_data, aes(x = Global_Time, y = DA_Identity, color = Subgroup)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_smooth(method = "loess", se = FALSE, aes(group = Subgroup), size=1.5) +
  scale_color_manual(values = my_cols) +
  theme_minimal() +
  labs(title = "Loss of Neuronal Identity",
       subtitle = "Global Pseudotime vs. Functional Identity",
       x = "Disease Trajectory (Pseudotime)",
       y = "Dopaminergic Identity Score")

# PLOT: Universal Stress
p2 <- ggplot(plot_data, aes(x = Global_Time, y = General_Stress, color = Subgroup)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_smooth(method = "loess", se = FALSE, aes(group = Subgroup), size=1.5) +
  scale_color_manual(values = my_cols) +
  theme_minimal() +
  labs(title = "Universal Cellular Stress",
       subtitle = "Global Pseudotime vs. Stress Response",
       x = "Disease Trajectory (Pseudotime)",
       y = "General Stress Score")

ggsave("Pseudotime_vs_Identity.png", plot = p1, width = 10, height = 7)
ggsave("Pseudotime_vs_Stress.png", plot = p2, width = 10, height = 7)

print("Done! Check 'Pseudotime_vs_Identity.png' and 'Pseudotime_vs_Stress.png'")
