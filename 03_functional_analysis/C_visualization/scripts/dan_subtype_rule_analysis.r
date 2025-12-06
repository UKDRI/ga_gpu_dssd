library(Seurat)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(stringr)

# ==========================================
# 1. Setup and Data Loading
# ==========================================

# Paths (Adjust if necessary)
seurat_path <- file.path("sc_data", "SNatlas_DaNs_seurat.RData")
binary_dir <- "sc_data"
# If auto-detection fails, set this to the exact column name (e.g., "cell_type_fine")
manual_column_name <- "CellSubType"

message("Loading Seurat Object...")
env <- new.env()
load(seurat_path, envir = env)
seurat_obj_name <- ls(env)[1]
seurat_obj <- env[[seurat_obj_name]]
message(paste("Loaded Seurat object:", seurat_obj_name))

message("Loading Binary Data...")

# Helper to read and standardize
read_data_file <- function(filename) {
  path <- file.path(binary_dir, filename)
  if(!file.exists(path)) stop(paste("File not found:", path))
  read_csv(path, show_col_types = FALSE)
}

X_binary <- read_data_file("X_binary.csv")
cell_ids <- read_data_file("cell_ids.csv")
y_labels <- read_data_file("y_labels.csv")

# Diagnostic prints
message("Dimensions Loaded:")
message(paste("X_binary:", paste(dim(X_binary), collapse = " x ")))
message(paste("cell_ids:", paste(dim(cell_ids), collapse = " x ")))
message(paste("y_labels:", paste(dim(y_labels), collapse = " x ")))

# Standardization: Ensure cell_ids has a column named 'cell_id'
if(ncol(cell_ids) == 1) {
  colnames(cell_ids) <- "cell_id"
}

# Standardization: Ensure y_labels has a column named 'label'
if(ncol(y_labels) == 1) {
  colnames(y_labels) <- "label"
}

# Check dimensions again
if(nrow(X_binary) != nrow(cell_ids)) {
  stop(paste("Error: Row mismatch. X_binary has", nrow(X_binary), "rows, cell_ids has", nrow(cell_ids), "rows."))
}

# Combine binary data into one dataframe
binary_df <- bind_cols(cell_ids, y_labels, X_binary)

# ==========================================
# 2. Extract Metadata from Seurat
# ==========================================

message("Extracting DaN labels from Seurat...")

# Fetch metadata
meta <- seurat_obj@meta.data
meta$cell_id <- rownames(meta)

# Debug: Print available columns to help user identify the right one
message("--- Metadata Columns Available ---")
print(colnames(meta))
message("----------------------------------")

# --- NEW: Check all Disease types in the original object ---
if("Disease" %in% colnames(meta)) {
  message("\n--- Full Breakdown of Disease Labels in Seurat Object ---")
  print(table(meta$Disease))
  message("---------------------------------------------------------\n")
}

if (!is.null(manual_column_name) && manual_column_name %in% colnames(meta)) {
  message(paste("Using manually specified column:", manual_column_name))
  meta$DaN_Label <- meta[[manual_column_name]]
} else {
  # Heuristic: Find columns with "DaN" in the name OR in the values
  candidate_cols <- names(meta)[sapply(meta, function(x) any(grepl("DaN", x, ignore.case = TRUE)))]

  if (length(candidate_cols) > 0) {
    message(paste("Found columns containing 'DaN':", paste(candidate_cols, collapse = ", ")))

    # Selection Logic: Pick the column with the MOST unique values.
    # (Assuming subtypes "Dan0, Dan1..." have more levels than major type "DaN")
    unique_counts <- sapply(candidate_cols, function(col) length(unique(meta[[col]])))
    best_col <- names(which.max(unique_counts))

    message(paste("Automatically selecting column with most detail:", best_col, "(", unique_counts[best_col], "unique values )"))
    meta$DaN_Label <- meta[[best_col]]
  } else {
    warning("Could not automatically find a column with 'DaN' labels. Defaulting to active Idents.")
    meta$DaN_Label <- Idents(seurat_obj)
  }
}

# Check for Disease column to verify mapping
cols_to_select <- c("cell_id", "DaN_Label")
if("Disease" %in% colnames(meta)) {
  cols_to_select <- c(cols_to_select, "Disease")
}

# Merge Seurat DaN labels into Binary DF
full_data <- inner_join(binary_df, meta %>% select(all_of(cols_to_select)), by = "cell_id")
message(paste("Matched", nrow(full_data), "cells between binary data and Seurat object."))

# --- VERIFICATION STEP ---
if("Disease" %in% colnames(full_data)) {
  message("\n==============================================")
  message("CHECK: Mapping between Binary Labels (0/1) and Seurat Disease")
  message("==============================================")
  print(table(Binary_Label = full_data$label, Seurat_Disease = full_data$Disease))
  message("==============================================\n")
}

# Convert labels to readable factors
if(is.numeric(full_data$label)) {
  full_data$Condition <- ifelse(full_data$label == 1, "PD", "Control")
} else {
  full_data$Condition <- full_data$label
}

# ==========================================
# 3. Define and Apply Rules + Stats
# ==========================================

# Helper function to apply a logical filter and return summary
analyze_rule <- function(data, rule_expr, rule_name) {

  # Filter cells matching the rule
  subset_df <- data %>% filter(eval(parse(text = rule_expr)))

  # --- PRINT TOTAL COVERAGE ---
  total_cells <- nrow(subset_df)
  message(paste("\n------------------------------------------------"))
  message(paste("ANALYZING:", rule_name))
  message(paste("Total cells matching rule:", total_cells))
  message(paste("------------------------------------------------"))

  if(total_cells == 0) {
    return(NULL)
  }

  # --- STATISTICAL TEST ---
  # Create contingency table for this rule: Condition vs DaN Type
  cont_table <- table(subset_df$Condition, subset_df$DaN_Label)

  message("Contingency Table (Condition vs Subtype):")
  print(cont_table)

  # Check specifically for DaN_3 presence
  if("DaN_3" %in% colnames(cont_table)) {
    dan3_counts <- cont_table[, "DaN_3"]
    message("DaN_3 Counts:")
    print(dan3_counts)
    if(sum(dan3_counts) > 0 && any(dan3_counts == 0)) {
      message("NOTE: DaN_3 appears to be exclusive to one condition in this rule.")
    }
  } else {
    message("No DaN_3 cells found in this rule subset.")
  }

  # Fisher's Exact Test (Robust for small numbers)
  # We simulate p-value if workspace is too large, otherwise exact
  if(nrow(cont_table) > 1 && ncol(cont_table) > 1) {
    test_res <- tryCatch({
      fisher.test(cont_table, workspace = 2e8)
    }, error = function(e) {
      fisher.test(cont_table, simulate.p.value = TRUE)
    })
    message(paste("Fisher's Exact Test p-value:", format.pval(test_res$p.value, digits=4)))
  } else {
    message("Not enough dimensions for Fisher's test (e.g., only one condition or subtype present).")
  }

  # Calculate proportions
  summary_stats <- subset_df %>%
    group_by(Condition, DaN_Label) %>%
    summarise(Count = n(), .groups = "drop") %>%
    group_by(Condition) %>%
    mutate(
      Total_In_Condition = sum(Count),
      Proportion = Count / Total_In_Condition,
      Rule = rule_name
    )

  return(summary_stats)
}

# Define Rules
rules <- list(
  "Rule 1" = "NCOR2==0 & FAM19A5==0 & BRSK2==0 & `MT-ND4L`==1 & BIN1==0 & SEMA4D==0 & FAM102A==0 & ZFYVE16==0 & ST3GAL4==0 & ATP6V1D==0 & DSTN==0 & CHST3==0 & PALD1==0 & COX4I1==0 & FNTA==0",
  "Rule 2" = "FTH1==1 & RBX1==0 & PALD1==0 & SLTM==0 & TMEM219==0",
  "Rule 3" = "RPL21==1 & PDE2A==0 & AGAP3==0 & WNT7B==0 & SELENOT==0 & TMEM201==1 & MUC3A==0 & DAD1==1 & RET==1 & PPT1==0 & SPART==0 & SLIRP==1 & VAPA==0 & RHPN1==0 & YWHAB==0 & PCDH17==0 & RPS12==0",
  "Rule 4" = "PALD1==1"
)

results_list <- list()

message("Applying rules...")
for(r_name in names(rules)) {
  # Message handled inside function now
  tryCatch({
    res <- analyze_rule(full_data, rules[[r_name]], r_name)
    results_list[[r_name]] <- res
  }, error = function(e) {
    message(paste("Error in", r_name, ":", e$message))
  })
}

final_results <- bind_rows(results_list)

# ==========================================
# 4. Visualization
# ==========================================

if(nrow(final_results) > 0) {

  final_results$Rule <- factor(final_results$Rule, levels = names(rules))

  p <- ggplot(final_results, aes(x = Condition, y = Proportion, fill = DaN_Label)) +
    geom_bar(stat = "identity", position = "fill") +
    facet_wrap(~Rule, scales = "free_x") +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = "DaN Subtype Composition by Rule and Condition",
      y = "Proportion",
      x = "Condition",
      fill = "DaN Subtype"
    ) +
    theme_minimal() +
    theme(
      strip.background = element_rect(fill = "lightgrey", color = NA),
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(face = "bold")
    )

  ggsave("DaN_Composition_by_Rule.png", p, width = 10, height = 8)
  print(p)
  print(final_results %>% select(Rule, Condition, DaN_Label, Count, Proportion) %>% arrange(Rule, Condition))

} else {
  message("No cells matched any rules.")
}
