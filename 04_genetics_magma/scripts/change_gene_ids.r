# install.packages("BiocManager")
# BiocManager::install("org.Hs.eg.db")

library(org.Hs.eg.db)

# Path to your symbol-based sets
in_file  <- "/Users/samneaves/Documents/ga_gpu_dssd/magma/subgroups.set"
out_file <- "/Users/samneaves/Documents/ga_gpu_dssd/magma/subgroups_entrez.set"

# Read as plain text, one line per rule
lines <- readLines(in_file)

convert_line <- function(line) {
  parts <- strsplit(line, "\\s+")[[1]]
  set_name <- parts[1]
  genes_sym <- parts[-1]

  # Map symbols -> Entrez IDs
  entrez <- mapIds(
    org.Hs.eg.db,
    keys = genes_sym,
    keytype = "SYMBOL",
    column = "ENTREZID",
    multiVals = "first"
  )

  # Drop NAs (symbols with no Entrez mapping)
  entrez <- entrez[!is.na(entrez)]

  if (length(entrez) == 0) {
    warning(sprintf("Set %s has no mapped Entrez IDs", set_name))
  }

  paste(c(set_name, unname(entrez)), collapse = "\t")
}

converted <- vapply(lines, convert_line, character(1L))
writeLines(converted, out_file)
