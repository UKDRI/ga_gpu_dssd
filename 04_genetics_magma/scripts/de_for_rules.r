#!/usr/bin/env Rscript

## ------------------------------------------------------------------
## Differential expression for GA-GPU rules + MAGMA gene-set creation
## ------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(org.Hs.eg.db)
})

## -----------------------------
## Paths (edit if you move files)
## -----------------------------
sc_dir    <- "/Users/samneaves/Documents/ga_gpu_dssd/sc_data"
magma_dir <- "/Users/samneaves/Documents/ga_gpu_dssd/magma"

seurat_file   <- file.path(sc_dir, "SNatlas_DaNs_seurat.RData")
x_binary_file <- file.path(sc_dir, "X_binary.csv")
cell_ids_file <- file.path(sc_dir, "cell_ids.csv")

setwd(magma_dir)

cat("Loading Seurat object from:", seurat_file, "\n")
load(seurat_file)

## Try to detect the Seurat object in the .RData file
objs <- ls()
seurat_obj_name <- NULL
for (obj in objs) {
  if (inherits(get(obj), "Seurat")) {
    seurat_obj_name <- obj
    break
  }
}
if (is.null(seurat_obj_name)) {
  stop("No Seurat object found in SNatlas_DaNs_seurat.RData.")
}
cat("Using Seurat object:", seurat_obj_name, "\n")
seu <- get(seurat_obj_name)

## -------------------------
## Create PD / CTRL metadata
## -------------------------

if (!"Disease" %in% colnames(seu@meta.data)) {
  stop("Metadata column 'Disease' not found in Seurat object.")
}

seu$PD_status <- NA_character_
seu$PD_status[seu$Disease == "CTR"] <- "CTRL"
seu$PD_status[grepl("^PD", seu$Disease)] <- "PD"

table_PD_status <- table(seu$PD_status, useNA = "ifany")
cat("PD_status counts:\n")
print(table_PD_status)

## Keep only cells with known PD/CTRL status
valid_cells <- colnames(seu)[!is.na(seu$PD_status)]
seu <- subset(seu, cells = valid_cells)

## ------------------------
## Load binary matrix + IDs
## ------------------------

cat("Loading binary matrix from:", x_binary_file, "\n")
X_bin <- read.csv(x_binary_file, header = TRUE, check.names = FALSE)
X_bin <- as.matrix(X_bin)

cat("Loading cell IDs from:", cell_ids_file, "\n")
cell_ids_df <- read.csv(cell_ids_file, header = TRUE, stringsAsFactors = FALSE)
if (ncol(cell_ids_df) == 1) {
  cell_ids <- cell_ids_df[[1]]
} else if ("cell_id" %in% names(cell_ids_df)) {
  cell_ids <- cell_ids_df$cell_id
} else {
  ## fallback: first column
  cell_ids <- cell_ids_df[[1]]
}

if (nrow(X_bin) != length(cell_ids)) {
  stop("Row count of X_binary.csv (", nrow(X_bin),
       ") does not match length of cell_ids.csv (", length(cell_ids), ").")
}
rownames(X_bin) <- cell_ids

## Only keep cells that are actually in the Seurat object
common_cells <- intersect(rownames(X_bin), colnames(seu))
if (length(common_cells) == 0) {
  stop("No overlap between cells in X_binary and Seurat object.")
}
X_bin <- X_bin[common_cells, , drop = FALSE]
seu   <- subset(seu, cells = common_cells)

cat("Number of cells after intersecting binary + Seurat:", ncol(seu), "\n")

## -----------------------
## Helper: rule evaluation
## -----------------------

get_rule_cells <- function(mat, expr) {
  idx <- eval(expr, envir = as.data.frame(mat))
  rownames(mat)[which(idx)]
}

## Ensure the required genes exist
required_genes <- c(
  # Rule1
  "NCOR2","FAM19A5","BRSK2","MT-ND4L","BIN1","SEMA4D","FAM102A",
  "ZFYVE16","ST3GAL4","ATP6V1D","DSTN","CHST3","PALD1","COX4I1","FNTA",
  # Rule2
  "FTH1","RBX1","SLTM","TMEM219",
  # Rule3 (for completeness, though we won't DE it)
  "RPL21","PDE2A","AGAP3","WNT7B","SELENOT","TMEM201","MUC3A","DAD1",
  "RET","PPT1","SPART","SLIRP","VAPA","RHPN1","YWHAB","PCDH17","RPS12",
  # Rule4
  "PALD1"
)

missing <- setdiff(required_genes, colnames(X_bin))
if (length(missing) > 0) {
  warning("The following rule genes are missing from X_binary: ",
          paste(missing, collapse = ", "))
}

## -------------------------
## Evaluate the four rules
## -------------------------

# Coerce to data.frame to avoid factor issues
X_df <- as.data.frame(X_bin)

rule1_expr <- quote(
  NCOR2 == 0 & FAM19A5 == 0 & BRSK2 == 0 & `MT-ND4L` == 1 & BIN1 == 0 &
  SEMA4D == 0 & FAM102A == 0 & ZFYVE16 == 0 & ST3GAL4 == 0 & ATP6V1D == 0 &
  DSTN == 0 & CHST3 == 0 & PALD1 == 0 & COX4I1 == 0 & FNTA == 0
)

rule2_expr <- quote(
  FTH1 == 1 & RBX1 == 0 & PALD1 == 0 & SLTM == 0 & TMEM219 == 0
)

rule3_expr <- quote(
  RPL21 == 1 & PDE2A == 0 & AGAP3 == 0 & WNT7B == 0 & SELENOT == 0 &
  TMEM201 == 1 & MUC3A == 0 & DAD1 == 1 & RET == 1 & PPT1 == 0 &
  SPART == 0 & SLIRP == 1 & VAPA == 0 & RHPN1 == 0 & YWHAB == 0 &
  PCDH17 == 0 & RPS12 == 0
)

rule4_expr <- quote(
  PALD1 == 1
)

rule1_cells <- get_rule_cells(X_df, rule1_expr)
rule2_cells <- get_rule_cells(X_df, rule2_expr)
rule3_cells <- get_rule_cells(X_df, rule3_expr)
rule4_cells <- get_rule_cells(X_df, rule4_expr)

cat("Rule1 cells:", length(rule1_cells), "\n")
cat("Rule2 cells:", length(rule2_cells), "\n")
cat("Rule3 cells:", length(rule3_cells), "\n")
cat("Rule4 cells:", length(rule4_cells), "\n")

## ---------------------------------------------
## Helper: run DE + build Entrez gene-set lines
## ---------------------------------------------

run_DE_for_rule <- function(seu, rule_name, rule_cells,
                            min_cells_per_group = 30,
                            top_n = 300) {
  rule_cells <- intersect(rule_cells, colnames(seu))
  message("---- ", rule_name, " ----")
  message("Total rule cells: ", length(rule_cells))

  ## ---------- PD vs CTRL within rule ----------
  de_PDvsCTRL <- NULL

  if (length(rule_cells) >= (2 * min_cells_per_group)) {
    pd_cells   <- rule_cells[seu$PD_status[rule_cells] == "PD"]
    ctrl_cells <- rule_cells[seu$PD_status[rule_cells] == "CTRL"]

    message("PD_rule cells: ", length(pd_cells),
            " | CTRL_rule cells: ", length(ctrl_cells))

    if (length(pd_cells) >= min_cells_per_group &&
        length(ctrl_cells) >= min_cells_per_group) {

      message("Running PD vs CTRL DE for ", rule_name)

      sub_obj <- subset(seu, cells = c(pd_cells, ctrl_cells))
      sub_obj$Group <- sub_obj$PD_status
      Idents(sub_obj) <- sub_obj$Group

      de_PDvsCTRL <- FindMarkers(
        sub_obj,
        ident.1 = "PD",
        ident.2 = "CTRL",
        test.use = "MAST",
        logfc.threshold = 0.25,
        min.pct = 0.1
      )

    } else {
      message("Skipping ", rule_name,
              " PD_vs_CTRL: PD or CTRL count below threshold (",
              "PD=", length(pd_cells),
              ", CTRL=", length(ctrl_cells), ")")
    }
  } else {
    message("Skipping ", rule_name,
            " PD_vs_CTRL: not enough total rule cells.")
  }

  ## ---------- Rule vs non-rule ----------
  de_rule_vs_rest <- NULL

  nonrule_cells <- setdiff(colnames(seu), rule_cells)
  message("Non-rule cells: ", length(nonrule_cells))

  if (length(rule_cells) >= min_cells_per_group &&
      length(nonrule_cells) >= min_cells_per_group) {

    message("Running Rule vs non-rule DE for ", rule_name)

    sub_obj2 <- subset(seu, cells = c(rule_cells, nonrule_cells))
    sub_obj2$RuleFlag <- ifelse(colnames(sub_obj2) %in% rule_cells,
                                "Rule", "Rest")
    Idents(sub_obj2) <- sub_obj2$RuleFlag

    de_rule_vs_rest <- FindMarkers(
      sub_obj2,
      ident.1 = "Rule",
      ident.2 = "Rest",
      test.use = "MAST",
      logfc.threshold = 0.25,
      min.pct = 0.1
    )

  } else {
    message("Skipping ", rule_name,
            " rule_vs_rest: not enough cells in one of the groups.")
  }

  list(
    PDvsCTRL   = de_PDvsCTRL,
    RuleVsRest = de_rule_vs_rest
  )
}

## --------------
## Run DE by rule
## --------------

DefaultAssay(seu) <- "RNA"

de_rule1 <- run_DE_for_rule(seu, "Rule1", rule1_cells)
de_rule2 <- run_DE_for_rule(seu, "Rule2", rule2_cells)
# Rule3 has only 1 cell; likely skipped
de_rule3 <- run_DE_for_rule(seu, "Rule3", rule3_cells)
de_rule4 <- run_DE_for_rule(seu, "Rule4", rule4_cells)

de_list <- list(Rule1 = de_rule1, Rule2 = de_rule2, Rule3 = de_rule3, Rule4 = de_rule4)

## --------------------------------------
## Helper: DE → top Entrez IDs → .set line
## --------------------------------------

de_to_entrez_line <- function(de_res, set_name, top_n = 300,
                              p_adj_cutoff = 0.05) {
  if (is.null(de_res) || nrow(de_res) == 0) return(NULL)

  de_res <- de_res %>%
    mutate(gene = rownames(de_res)) %>%
    arrange(p_val_adj)

  de_res <- de_res %>%
    filter(!is.na(p_val_adj), p_val_adj < p_adj_cutoff)

  if (nrow(de_res) == 0) return(NULL)

  de_res <- head(de_res, top_n)

  symbols <- de_res$gene
  entrez <- mapIds(
    org.Hs.eg.db,
    keys = symbols,
    keytype = "SYMBOL",
    column = "ENTREZID",
    multiVals = "first"
  )

  entrez <- unique(na.omit(entrez))
  if (length(entrez) == 0) return(NULL)

  paste(c(set_name, entrez), collapse = "\t")
}

## ------------------------------------
## Build MAGMA gene-set files (.set)
## ------------------------------------

pdctrl_lines    <- c()
rule_rest_lines <- c()

for (rname in names(de_list)) {
  dl <- de_list[[rname]]

  ## PD vs CTRL within rule
  if (!is.null(dl$PDvsCTRL)) {
    line <- de_to_entrez_line(
      dl$PDvsCTRL,
      set_name = paste0(rname, "_PDvsCTRL"),
      top_n = 300
    )
    if (!is.null(line)) {
      pdctrl_lines <- c(pdctrl_lines, line)
    } else {
      message("No PD_vs_CTRL genes passed filters for ", rname)
    }
  } else {
    message("No PD_vs_CTRL DE result for ", rname)
  }

  ## Rule vs non-rule
  if (!is.null(dl$RuleVsRest)) {
    line2 <- de_to_entrez_line(
      dl$RuleVsRest,
      set_name = paste0(rname, "_RuleVsRest"),
      top_n = 300
    )
    if (!is.null(line2)) {
      rule_rest_lines <- c(rule_rest_lines, line2)
    } else {
      message("No Rule_vsRest genes passed filters for ", rname)
    }
  } else {
    message("No Rule_vsRest DE result for ", rname)
  }
}

pdctrl_file    <- file.path(magma_dir, "rule_PDvsCTRL_entrez.set")
rule_rest_file <- file.path(magma_dir, "rule_ruleVsRest_entrez.set")

if (length(pdctrl_lines) > 0) {
  writeLines(pdctrl_lines, pdctrl_file)
  cat("Wrote PD_vs_CTRL gene sets to:", pdctrl_file, "\n")
} else {
  cat("No PD_vs_CTRL gene sets produced (check DE thresholds).\n")
}

if (length(rule_rest_lines) > 0) {
  writeLines(rule_rest_lines, rule_rest_file)
  cat("Wrote rule_vs_rest gene sets to:", rule_rest_file, "\n")
} else {
  cat("No rule_vs_rest gene sets produced (check DE thresholds).\n")
}

cat("Done.\n")
