#env scanpy_scimilarity

print("Starting")

import os
import numpy as np
import anndata as ad
import scanpy as sc
import scimilarity as scim
from scimilarity.utils import align_dataset, lognorm_counts
from scimilarity.cell_embedding import CellEmbedding
from tqdm.auto import tqdm

print("Imports Complete")

# Model from https://zenodo.org/records/10685499
ce = CellEmbedding(model_path="/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/scPortrait_vit_250726/SCimilarity/models/model_v1.1")

#log1p normalized filtered xenium
data_in = ad.read_h5ad("/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/scPortrait_vit_250726/Data/xenium_ovarian_cancer_vitmae_feats.h5ad")

# # Scale mean of counts per cell to 10000 to preserve relative differences (In Xenium the rel diffs are not stochasic or related to seq depth)
# current_mean = np.array(data_in.X.sum(axis=1)).mean()
# # scale_factor = 1e4 / current_mean
# # Scale factor calculated on filtered transcriptome. This is important because it excludes very highly expressed genes, affecting scaling
# scale_factor = np.float32(39.682293)
# data_in.X = data_in.X.multiply(scale_factor)

# sc.pp.log1p(data_in)

sc.pp.normalize_total(data_in, target_sum=1e4)
# X = data_in.X
# X = X / X.max() * np.log(10001)
# data_in.X = X
# SCimilarity value range: 0 - 9.21044036698 (ln(1) - ln(10001))
# data_in = ad.read_h5ad("/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/lamin_xenium/Data/filtered_transcriptome.h5ad")

# cell_labels = np.load("/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/lamin_xenium/Data/matched_vitmae_transcripts/cell_labels.npy")
# # Harmonize to data saved from main xenium notebook. Those data were used for training the MLPs
# data_in = data_in[data_in.obs['cell_labels'].isin(cell_labels)]
# np.save("/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/lamin_xenium/Data/matched_vitmae_transcripts/scimilarity_cell_labels_v1_1.npy", data_in.obs['cell_labels'])

print("Data Loaded")

# Have to lower gene_overlap because Xenium is targeted. Only has 3700 (really, that many?) genes in total. Default value is 5000
data_out = align_dataset(data_in, ce.gene_order, gene_overlap_threshold=1000)

print("Data Aligned")

embeds = ce.get_embeddings(data_out.X)

print("Got Embeddings")

np.save("/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/scPortrait_vit_250726/Data/scimilarity_v1_1_embeds_full_transcriptome_norm10k.npy", embeds)

print("Done")
