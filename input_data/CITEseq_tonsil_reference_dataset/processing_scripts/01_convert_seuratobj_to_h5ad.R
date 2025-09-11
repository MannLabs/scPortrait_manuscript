options(repos = c(CRAN = "https://cloud.r-project.org"))

# install Seurat from CRAN
if (!requireNamespace("Seurat", quietly = TRUE)) {
  install.packages("Seurat")
}

# GitHub: SeuratDisk
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
  remotes::install_github("mojaveazure/seurat-disk", upgrade = "never")
}

library("Seurat")
library("SeuratDisk")

object_path = "../raw_input_data/CITE-seq/20220215_tonsil_atlas_cite_seurat_obj.rds"
output_directory = "../processed_data"

 #create output directory if it does not exist
if (!dir.exists(output_directory)){
  dir.create(output_directory, recursive = TRUE)
}

# load the seurat object
cite_seurat <- readRDS(object_path)

# save the RNA assay information
SaveH5Seurat(cite_seurat, filename = "../processed_data/CITEseq_RNA.h5seurat")
Convert("../processed_data/CITEseq_RNA.h5seurat", dest = "h5ad", overwrite = FALSE)

# save the ADT assay information
DefaultAssay(cite_seurat) <- "ADT"  # get the CITEsqe
SaveH5Seurat(cite_seurat, filename = "../processed_data/CITEseq_PROT.h5seurat")
Convert("../processed_data/CITEseq_PROT.h5seurat", dest = "h5ad", overwrite = TRUE)
