# this script downloads the data for the following dataset:

# Xenium Prime 5K In Situ Gene Expression with Cell Segmentation data for human ovarian cancer (FFPE) using the Xenium Prime 5K Human Pan Tissue and Pathways Panel plus 100 Custom Genes.

# https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer

# If using this dataset please cite as:

# Xenium Prime 5K In Situ Gene Expression with Cell Segmentation data for human ovarian cancer (FFPE) using the Xenium Prime 5K Human Pan Tissue and Pathways Panel plus 100 Custom Genes (v1), In Situ Gene Expression dataset analyzed using Xenium Onboard Analysis 3.0.0, 10x Genomics, (2024, Dec 17).

# create subfolder for raw input data
mkdir -p raw_input_data

# download all required input files into raw_input_data
wget -P raw_input_data https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_gene_groups.csv
wget -P raw_input_data https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_cell_groups.csv
wget -P raw_input_data https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_imagealignment.csv
wget -P raw_input_data https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip
wget -P raw_input_data https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif
wget -P raw_input_data https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_annotated_image.ome.tif

# unzip input files inside raw_input_data
bsdtar -xf raw_input_data/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip -C raw_input_data && rm raw_input_data/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip