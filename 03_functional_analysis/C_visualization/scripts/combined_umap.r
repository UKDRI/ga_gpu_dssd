library(Seurat)
library(ggplot2)
library(dplyr)

# 1. Load Base Object
data_file <- "./sc_data/SNatlas_DaNs_seurat.RData"
print("Loading base Seurat object...")
load(data_file)
# Handle dynamic object name
obj_list <- ls()
seurat_obj_name <- obj_list[obj_list != "obj_list" & obj_list != "data_file"]
seurat_obj <- get(seurat_obj_name[1])

# 2. Load Rule Data
print("Loading CSVs...")
X_bin <- read.csv("./sc_data/X_binary.csv", check.names=FALSE) # check.names=FALSE keeps 'MT-ND4L'
cell_ids <- read.csv("./sc_data/cell_ids.csv")
actual_ids <- cell_ids[,1]

# 3. Apply Rules to Create Masks
print("Applying Rules...")

# Rule 1: Metabolic (MT-ND4L=1, etc)
r1_mask <- with(X_bin, `MT-ND4L` == 1 & NCOR2 == 0 & FAM19A5 == 0 & BRSK2 == 0 &
                  BIN1 == 0 & SEMA4D == 0 & FAM102A == 0 & ZFYVE16 == 0 &
                  ST3GAL4 == 0 & ATP6V1D == 0 & DSTN == 0 & CHST3 == 0 &
                  PALD1 == 0 & COX4I1 == 0 & FNTA == 0)

# Rule 2: Iron/Stress (FTH1=1, etc)
r2_mask <- with(X_bin, FTH1 == 1 & RBX1 == 0 & PALD1 == 0 & SLTM == 0 & TMEM219 == 0)

# Rule 4: Regeneration (PALD1=1)
r4_mask <- with(X_bin, PALD1 == 1)

# Get Barcodes for each group
r1_barcodes <- actual_ids[r1_mask]
r2_barcodes <- actual_ids[r2_mask]
r4_barcodes <- actual_ids[r4_mask]

print(paste("Rule 1 Count:", length(r1_barcodes)))
print(paste("Rule 2 Count:", length(r2_barcodes)))
print(paste("Rule 4 Count:", length(r4_barcodes)))

# 4. Filter Seurat Object
# Filter to only the cells we analyzed (Binary Matrix rows)
# AND keep only PD donors (Assuming 'Disease' column exists and CTR is control)
# Adjust "CTR" or "Control" to match your actual metadata
seurat_subset <- subset(seurat_obj, cells = actual_ids)
control_labels <- c("HC", "Control", "CTR", "Healthy", "HC1")
seurat_subset <- subset(seurat_subset, subset = !Disease %in% control_labels)

# 5. Add Combined Metadata
# Default is "Other"
seurat_subset$Subgroup_Status <- "Other"

# Assign Rules (Order matters if there's overlap, but Venn said overlap is minimal)
seurat_subset$Subgroup_Status[colnames(seurat_subset) %in% r1_barcodes] <- "Rule 1 (Metabolic)"
seurat_subset$Subgroup_Status[colnames(seurat_subset) %in% r2_barcodes] <- "Rule 2 (Iron)"
seurat_subset$Subgroup_Status[colnames(seurat_subset) %in% r4_barcodes] <- "Rule 4 (Regeneration)"

# Set Order so "Other" is plotted first (at the back), and colored points on top
seurat_subset$Subgroup_Status <- factor(seurat_subset$Subgroup_Status,
                                        levels = c("Other", "Rule 1 (Metabolic)", "Rule 2 (Iron)", "Rule 4 (Regeneration)"))

# 6. PLOT: Combined Confetti UMAP
print("Plotting UMAP...")

# Define Colors to match your other plots
# Other = Light Grey, R1 = Blue, R2 = Orange, R4 = Green
my_cols <- c("Other" = "#e0e0e0",
             "Rule 1 (Metabolic)" = "#4c72b0",
             "Rule 2 (Iron)" = "#dd8452",
             "Rule 4 (Regeneration)" = "#55a868")

p1 <- DimPlot(seurat_subset, group.by = "Subgroup_Status",
              cols = my_cols,
              order = TRUE,  # Forces colored cells to be plotted ON TOP of grey cells
              pt.size = 0.8) +
      ggtitle("Functional States vs Global Clusters") +
      theme(legend.position = "bottom")

ggsave("combined_umap_confetti.png", plot = p1, width = 7, height = 7)

print("Success! Check 'combined_umap_confetti.png'")
