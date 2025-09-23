# CODEX data
mkdir -p input_data_CODEX
wget -O input_data_CODEX/scportrait_project_codex_region1.zip "https://datashare.biochem.mpg.de/s/nmJRNeWoC92nZay/download"
bsdtar -xf input_data_CODEX/scportrait_project_codex_region1.zip -C raw_input_data && rm input_data_CODEX/scportrait_project_codex_region1.zip
wget -O input_data_CODEX/CITEseq_rna_processed_full_genome.h5ad "https://datashare.biochem.mpg.de/s/ayHoH4wFxJQ5a7J/download"

# Xenium data
mkdir -p input_data_Xenium
wget -O input_data_Xenium/xenium_scportrait_sdata.tar.gz "https://datashare.biochem.mpg.de/s/fQAYAk6Ze6aApiP/download"
tar -xf input_data_Xenium/xenium_scportrait_sdata.tar.gz -C input_data_Xenium && rm input_data_Xenium/xenium_scportrait_sdata.tar.gz
wget -O input_data_Xenium/xenium_ovarian_cancer_full.h5ad "https://datashare.biochem.mpg.de/s/RgTbmAMkYgKr9QL/download"
wget -O input_data_Xenium/single_cells.h5sc "https://datashare.biochem.mpg.de/s/ExiEzNBPp5fWkHL/download"
wget -O input_data_Xenium/all_cells_tsne_coordinates.csv "https://datashare.biochem.mpg.de/s/RRecfkfHACcyfW5/download"
wget -O input_data_Xenium/macrophage_image_leiden.csv "https://datashare.biochem.mpg.de/s/tYyjeimZodQHmjE/download"
wget -O input_data_Xenium/TestSet_Image_SCimilarity_embeddings_tsne_coords.npy "https://datashare.biochem.mpg.de/s/CwcCm2ABcoDSYrE/download"
wget -O input_data_Xenium/test_set_tsne_coordinates.csv "https://datashare.biochem.mpg.de/s/LGfaXcXFYCD8TKP/download"
wget -O input_data_Xenium/test_set_macs_clusters.csv "https://datashare.biochem.mpg.de/s/8YjTxiqQGL6RQjx/download"