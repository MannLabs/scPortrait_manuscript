#env SPAspa

print("Starting")

import sys
sys.path.append("model_archs")
import mlp_img_embed_to_scimilarity

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
# from torch.utils.data import DataLoader, random_split, TensorDataset
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar

print("Imports complete")

top_x_pcas = 10

print("Training to predict SCimilarity embeddings")

data_dir = "/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/scPortrait_vit_250726/Data/"
ckpt_dir = "/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/scPortrait_vit_250726/SCimilarity/training_mlp/checkpoints/predict_scim_v1_1/scaled_norm10k/all_cells/"

# scimilarity_model_path = '/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/lamin_xenium/SCimilarity/models/'
# out_dir = '/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/scPortrait_vit_250726/Data/'
# data_dir = '/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/lamin_xenium/Data/'
# sdata_path = '/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/lamin_xenium/Data/xenium_sdata/'

class MLP_vitmae_to_scimilarity(pl.LightningModule):
    def __init__(self):
        super(MLP_vitmae_to_scimilarity, self).__init__()
        self.model = mlp_img_embed_to_scimilarity.get_mlp(output_size=128)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # current_lr = self.optimizers().param_groups[0]['lr']
        # self.log("learning_rate", current_lr, logger=True)
        return loss
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)

class Embeddings_transcripts_dataset(Dataset):
    def __init__(self, embeddings, gene_expressions):
        """
        embeddings: (num_samples, embedding_dim)
        gene_expressions: (num_samples, num_genes)
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.gene_expressions = torch.tensor(gene_expressions, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.gene_expressions[idx]

train_set = ad.read_h5ad("/fs/gpfs41/lv03/fileset01/pool/pool-mann-maedler-shared/niklas_workspace/scPortrait_vit_250726/Data/xenium_ovarian_cancer_vitmae_feats.h5ad")
# img_features = img_features[img_features.obs['transcriptome_passed_QC'] == True]
# img_features = img_features[img_features.var['gene_passed_QC'] == True]
train_set = (train_set[~train_set.obs['is_in_vitmae_test_set']]) # & img_features.obs['transcriptome_passed_QC']

img_features = train_set.obsm['X_vitmae_finetuned_img_features_minmax']
scimilarity_embeddings = train_set.obsm['X_SCimilarity_transcriptome_embeds']

# img_features = np.load(f"{data_dir}matched_vitmae_transcripts/vitmae_features.npy")
# Normalize whole dataset
# img_features = np.interp(img_features, (img_features.min(), img_features.max()), (0, 1))

# # Normalize each row (used this, but now load minmax directly)
# img_features = (img_features - img_features.min(axis=1, keepdims=True)) / (
#     img_features.max(axis=1, keepdims=True) - img_features.min(axis=1, keepdims=True)
# )

# scimilarity_embeddings = np.load(f"{data_dir}scimilarity_v1_1_embeds_full_transcriptome_scaled39_log1p.npy")
# scimilarity_embeddings = scimilarity_embeddings[train_set]

# Column-wise-norm
scimilarity_embeddings, means, stds = mlp_img_embed_to_scimilarity.scale(scimilarity_embeddings)
# scimilarity_embeddings = np.load(f"{data_dir}matched_vitmae_transcripts/scimilarity_embeds_scpp_scaled.npy")
# Normalize whole dataset
# scimilarity_embeddings = np.interp(scimilarity_embeddings, (scimilarity_embeddings.min(), scimilarity_embeddings.max()), (0, 1))
# Normalize each row
# scimilarity_embeddings = (scimilarity_embeddings - scimilarity_embeddings.min(axis=1, keepdims=True)) / (
#     scimilarity_embeddings.max(axis=1, keepdims=True) - scimilarity_embeddings.min(axis=1, keepdims=True)
# )

np.save(f"{ckpt_dir}/means.npy", means)
np.save(f"{ckpt_dir}/stds.npy", stds)

dataset = Embeddings_transcripts_dataset(img_features, scimilarity_embeddings)

print("Data Loaded")

torch.manual_seed(920924)
num_samples = len(dataset)
train_size = int(0.9 * num_samples)
val_size = num_samples - train_size
# test_size = num_samples - train_size - val_size
train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, 0])

train_loader = DataLoader(train_dataset, batch_size=2048, num_workers=0, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=2048)

# print(next(iter(train_loader))[1][0])
# print(next(iter(train_loader))[1][0])
# print(next(iter(train_loader))[1][0])

model = MLP_vitmae_to_scimilarity()

checkpoint_callback = ModelCheckpoint(
    filename="ColNorm-{epoch}-end",
    dirpath=ckpt_dir,
    save_top_k=-1,
    every_n_epochs=1,
    save_on_train_epoch_end=True,
)

wandb_logger = WandbLogger(project="mlp-img-scim-v1.1")

trainer = pl.Trainer(
    max_epochs=10000,
    devices=[2],
    # strategy='ddp',
    accelerator="auto",
    logger=wandb_logger,
    check_val_every_n_epoch=1,
    # enable_progress_bar=False,
    callbacks=[
        checkpoint_callback,
    ],
    gradient_clip_val=0.5,
)

print("Starting Training")

trainer.fit(model, train_loader, val_loader)
