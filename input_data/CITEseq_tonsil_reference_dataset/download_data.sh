# this script downloads the data for the following dataset:

# CITEseq Seurat object associated with the tonsil cell atlas

# https://zenodo.org/records/8373756

# If using this dataset please cite as:

# Massoni-Badosa, R. (2022). Seurat objects associated with the tonsil cell atlas (2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8373756

# create subfolder for raw input data
mkdir -p raw_input_data

# download all required input files into raw_input_data
wget -O raw_input_data/TonsilAtlasSeuratCITE.tar.gz "https://zenodo.org/records/8373756/files/TonsilAtlasSeuratCITE.tar.gz?download=1"

# unzip input files inside raw_input_data
bsdtar -xzf raw_input_data/TonsilAtlasSeuratCITE.tar.gz -C raw_input_data && rm raw_input_data/TonsilAtlasSeuratCITE.tar.gz