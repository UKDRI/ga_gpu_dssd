#!/usr/bin/env Rscript

## ------------------------------------------------------------------
## TWAS Enrichment Analysis for GA-GPU Rules (Corrected)
## ------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(readr)
  library(stringr)
  library(knitr)
  library(utils)
})

## ------------------------------------------------------------------
## 1. Configuration & Paths
## ------------------------------------------------------------------

# User directories
base_dir  <- "/Users/samneaves/Documents/ga_gpu_dssd"
sc_dir    <- file.path(base_dir, "sc_data")
twas_dir  <- file.path(base_dir, "twas_ai")
out_dir   <- file.path(base_dir, "enrichment_results")

# Input Files
seurat_file   <- file.path(sc_dir, "SNatlas_DaNs_seurat.RData")
x_binary_file <- file.path(sc_dir, "X_binary.csv")
cell_ids_file <- file.path(sc_dir, "cell_ids.csv")

# TWAS Gene List Files
chatgpt_file  <- file.path(twas_dir, "chatgpt.txt")
gemini_file   <- file.path(twas_dir, "gemeni.csv")

# Create output directory
if (!dir.exists(out_dir)) dir.create(out_dir)

## ------------------------------------------------------------------
## 2. Load Data & Define Rules
## ------------------------------------------------------------------

message("Loading Seurat object...")
if (!file.exists(seurat_file)) stop("Seurat file not found: ", seurat_file)
load(seurat_file)

# Detect Seurat object name in the environment
objs <- ls()
seurat_obj_name <- NULL
for (obj in objs) {
  if (inherits(get(obj), "Seurat")) {
    seurat_obj_name <- obj
    break
  }
}
if (is.null(seurat_obj_name)) stop("No Seurat object found in RData file.")
seu <- get(seurat_obj_name)

# Ensure PD/CTRL metadata exists
if (!"Disease" %in% colnames(seu@meta.data)) stop("Metadata 'Disease' missing.")
seu$PD_status <- ifelse(grepl("^PD", seu$Disease), "PD",
                        ifelse(seu$Disease == "CTR", "CTRL", NA))
seu <- subset(seu, cells = colnames(seu))

# Load Binary Matrix for Rules
message("Loading Binary Matrix...")
if (!file.exists(x_binary_file)) stop("X_binary file not found.")
X_bin <- read.csv(x_binary_file, header = TRUE, check.names = FALSE)

if (!file.exists(cell_ids_file)) stop("Cell IDs file not found.")
cell_ids_df <- read.csv(cell_ids_file, header = TRUE, stringsAsFactors = FALSE)

# Handle potential column name variations for cell IDs
cid_col <- if("cell_id" %in% names(cell_ids_df)) "cell_id" else 1
rownames(X_bin) <- cell_ids_df[[cid_col]]

# Intersect cells
common_cells <- intersect(rownames(X_bin), colnames(seu))
if(length(common_cells) == 0) stop("No common cells between Seurat object and Binary matrix.")

X_bin <- X_bin[common_cells, ]
seu <- subset(seu, cells = common_cells)
X_df <- as.data.frame(X_bin)

# Define Rule Logic
get_rule_cells <- function(mat, expr) {
  tryCatch({
    idx <- eval(expr, envir = mat)
    rownames(mat)[which(idx)]
  }, error = function(e) { return(character(0)) })
}

rules <- list(
  Rule1 = quote(NCOR2==0 & FAM19A5==0 & BRSK2==0 & `MT-ND4L`==1 & BIN1==0 & SEMA4D==0 & FAM102A==0 & ZFYVE16==0 & ST3GAL4==0 & ATP6V1D==0 & DSTN==0 & CHST3==0 & PALD1==0 & COX4I1==0 & FNTA==0),
  Rule2 = quote(FTH1==1 & RBX1==0 & PALD1==0 & SLTM==0 & TMEM219==0),
  Rule3 = quote(RPL21==1 & PDE2A==0 & AGAP3==0 & WNT7B==0 & SELENOT==0 & TMEM201==1 & MUC3A==0 & DAD1==1 & RET==1 & PPT1==0 & SPART==0 & SLIRP==1 & VAPA==0 & RHPN1==0 & YWHAB==0 & PCDH17==0 & RPS12==0),
  Rule4 = quote(PALD1==1)
)

## ------------------------------------------------------------------
## 3. Parse TWAS Gene Lists (Corrected)
## ------------------------------------------------------------------

message("Parsing TWAS gene lists...")

# --- Parse ChatGPT (Markdown Table) ---
if (file.exists(chatgpt_file)) {
  lines <- readLines(chatgpt_file)
  # Filter for rows that look like table data: start with | but are not separator lines or headers
  data_lines <- lines

  chatgpt_genes <- sapply(data_lines, function(x) {
    # Split by pipe. The string "| Gene |" splits into "", " Gene ", ""
    parts <- str_split(x, "\\|")[[1]]
    if(length(parts) >= 2) {
      # Remove markdown bold (**), asterisks, and whitespace
      clean <- gsub("\\*|\\s+", "", parts[2])
      return(clean)
    } else {
      return(NA)
    }
  })
  chatgpt_genes <- unique(na.omit(chatgpt_genes))
  chatgpt_genes <- chatgpt_genes[chatgpt_genes!= ""]
  message(paste("Loaded", length(chatgpt_genes), "genes from ChatGPT list."))
} else {
  warning(paste("ChatGPT file not found at:", chatgpt_file))
  chatgpt_genes <- character(0)
}

# --- Parse Gemini (CSV) ---
if (file.exists(gemini_file)) {
  gemini_data <- read.csv(gemini_file, stringsAsFactors = FALSE)

  # Robust column checking
  # read.csv often converts spaces to dots (e.g., "Gene Symbol" -> "Gene.Symbol")
  target_col <- NULL

  if ("Gene.Symbol" %in% colnames(gemini_data)) {
    target_col <- "Gene.Symbol"
  } else if ("Gene Symbol" %in% colnames(gemini_data)) {
    target_col <- "Gene Symbol"
  } else if ("Gene" %in% colnames(gemini_data)) {
    target_col <- "Gene"
  }

  if (!is.null(target_col)) {
    gemini_genes <- gemini_data[[target_col]]
  } else {
    # Fallback: assume the first column is the gene symbol
    gemini_genes <- gemini_data[, 1]
  }

  gemini_genes <- unique(trimws(gemini_genes))
  gemini_genes <- gemini_genes[gemini_genes!= ""]
  message(paste("Loaded", length(gemini_genes), "genes from Gemini list."))
} else {
  warning(paste("Gemini file not found at:", gemini_file))
  gemini_genes <- character(0)
}

twas_sets <- list(ChatGPT = chatgpt_genes, Gemini = gemini_genes)

## ------------------------------------------------------------------
## 4. Run Differential Expression (DE)
## ------------------------------------------------------------------

# Function to run DE
run_de <- function(seurat_obj, cells_in_rule, rule_name) {
  res <- list(PDvsCTRL = NULL, RuleVsRest = NULL)

  # A. PD vs CTRL within Rule
  pd_cells <- intersect(cells_in_rule, colnames(seurat_obj))
  ctrl_cells <- intersect(cells_in_rule, colnames(seurat_obj))

  # Using a low threshold (3 cells) just to ensure code runs; adjust as needed for statistics
  if(length(pd_cells) >= 3 && length(ctrl_cells) >= 3) {
    message(paste("  Running PD vs CTRL for", rule_name))
    sub <- subset(seurat_obj, cells = c(pd_cells, ctrl_cells))
    Idents(sub) <- sub$PD_status
    try({
      res$PDvsCTRL <- FindMarkers(sub, ident.1 = "PD", ident.2 = "CTRL",
                                  test.use = "MAST", logfc.threshold = 0.25,
                                  min.pct = 0.1, verbose = FALSE)
    })
  }

  # B. Rule vs Rest
  rest_cells <- setdiff(colnames(seurat_obj), cells_in_rule)
  if(length(cells_in_rule) >= 3 && length(rest_cells) >= 3) {
    message(paste("  Running Rule vs Rest for", rule_name))
    seurat_obj$RuleGroup <- ifelse(colnames(seurat_obj) %in% cells_in_rule, "Rule", "Rest")
    Idents(seurat_obj) <- "RuleGroup"
    try({
      res$RuleVsRest <- FindMarkers(seurat_obj, ident.1 = "Rule", ident.2 = "Rest",
                                    test.use = "MAST", logfc.threshold = 0.25,
                                    min.pct = 0.1, verbose = FALSE)
    })
  }

  return(res)
}

# Execute DE Loop
de_results <- list()
DefaultAssay(seu) <- "RNA"

for(rname in names(rules)) {
  message(paste("Processing", rname, "..."))
  cells <- get_rule_cells(X_df, rules[[rname]])
  if(length(cells) > 0) {
    de_results[[rname]] <- run_de(seu, cells, rname)
  } else {
    message(paste("  No cells found for", rname))
  }
}

## ------------------------------------------------------------------
## 5. Enrichment Analysis (Fisher's Exact Test)
## ------------------------------------------------------------------

perform_enrichment <- function(de_table, rule, contrast, twas_set, twas_name, background) {
  if(is.null(de_table)) return(NULL)
  if(nrow(de_table) == 0) return(NULL)

  # Filter significant DE genes (P_adj < 0.05)
  sig_genes <- rownames(de_table)[de_table$p_val_adj < 0.05]
  sig_genes <- intersect(sig_genes, background)

  if(length(sig_genes) == 0) return(NULL)

  # Overlap
  target_genes <- intersect(twas_set, background)
  overlap <- intersect(sig_genes, target_genes)

  # Fisher Matrix
  #             In_TWAS   Not_In_TWAS
  # In_DE       a         b
  # Not_In_DE   c         d

  a <- length(overlap)
  b <- length(sig_genes) - a
  c <- length(target_genes) - a
  d <- length(background) - length(sig_genes) - c

  # Contingency table
  fisher_mat <- matrix(c(a, c, b, d), nrow=2)
  fisher <- fisher.test(fisher_mat, alternative = "greater") # Testing for enrichment

  return(data.frame(
    Rule = rule,
    Contrast = contrast,
    TWAS_List = twas_name,
    Overlap_Count = a,
    DE_Count = length(sig_genes),
    TWAS_Count = length(target_genes),
    P_Value = fisher$p.value,
    Odds_Ratio = as.numeric(fisher$estimate),
    Genes = paste(overlap, collapse = "; ")
  ))
}

# Run Enrichment Loop
background_genes <- rownames(seu)
final_results <- data.frame()

message("Running enrichment tests...")

for(rname in names(de_results)) {
  # Check PD vs CTRL
  res_pd <- de_results[[rname]]$PDvsCTRL
  if(!is.null(res_pd)) {
    for(tname in names(twas_sets)) {
      row <- perform_enrichment(res_pd, rname, "PD_vs_CTRL", twas_sets[[tname]], tname, background_genes)
      if(!is.null(row)) final_results <- rbind(final_results, row)
    }
  }

  # Check Rule vs Rest
  res_rule <- de_results[[rname]]$RuleVsRest
  if(!is.null(res_rule)) {
    for(tname in names(twas_sets)) {
      row <- perform_enrichment(res_rule, rname, "Rule_vs_Rest", twas_sets[[tname]], tname, background_genes)
      if(!is.null(row)) final_results <- rbind(final_results, row)
    }
  }
}

## ------------------------------------------------------------------
## 6. Save and Display Results
## ------------------------------------------------------------------

if(nrow(final_results) > 0) {
  # Sort by P-value
  final_results <- final_results %>% arrange(P_Value)

  # Print to console
  print(kable(final_results, digits = 4))

  # Save to CSV
  out_file <- file.path(out_dir, "TWAS_Enrichment_Results.csv")
  write.csv(final_results, out_file, row.names = FALSE)
  message(paste("Results saved to:", out_file))
} else {
  message("No significant enrichment found (or no significant DE genes to test against).")
}

message("Done.")
