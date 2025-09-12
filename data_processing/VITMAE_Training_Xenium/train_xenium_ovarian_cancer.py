import argparse
import sys
print(sys.executable)
import socket
print(socket.gethostname())
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb
import torch
import pandas as pd
torch.cuda.empty_cache()
import pytorch_lightning as pl
from pathlib import Path
from torchvision import transforms
from utils import feature_extractor
from models import TransformerModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from scportrait.tools.ml.datasets import H5ScSingleCellDataset
from scportrait.tools.ml.utils import split_dataset_fractions

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', type=int, default=1, help="Enable (1) or disable (0) finetuning")
    args = parser.parse_args()
    finetune = bool(args.finetune)
    # -------------------------------
    data_path = Path("scportrait_manuscript/input_data/Xenium_ovarian_cancer/processed_data/scPortrait_project_xenium/extraction/data/single_cells.h5sc")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(224),
        feature_extractor])
    layer_outputs = {}
    def get_intermediate(module, input, output):
        layer_outputs['encoder_layernorm'] = output
    savedir = '/lustre/groups/ml01/workspace/alioguz.can/scportrait4i/training_output'
    class_list = [0]
    return_id = True
    random_indices = False
    if random_indices:
        dataset = H5ScSingleCellDataset(dir_list=[data_path], dir_labels=class_list, select_channel=[2,3,4], transform=t, return_id=return_id)
        train_dataset, val_dataset, test_dataset = split_dataset_fractions(
            [dataset],
            fractions=[0.9, 0.05, 0.05],
            seed=42)
    else:
        ## init from predefined indices
        indices_folder = "scportrait_manuscript/input_data/Xenium_ovarian_cancer/processed_data/test_val_datasets"
        print(f"Reading indices from {indices_folder}")
        train_set_indexes = pd.read_csv(f'{indices_folder}/train_set_indexes.csv', header = None)[0].tolist()
        train_dataset = H5ScSingleCellDataset(dir_list=[data_path], dir_labels=class_list, select_channel=[2,3,4], transform=t, return_id=return_id, index_list=[train_set_indexes])

        test_set_indexes = pd.read_csv(f'{indices_folder}/test_set_indexes.csv', header = None)[0].tolist()
        test_dataset = H5ScSingleCellDataset(dir_list=[data_path], dir_labels=class_list, select_channel=[2,3,4], transform=t, return_id=return_id, index_list=[test_set_indexes])

        val_set_indexes = pd.read_csv(f'{indices_folder}/val_set_indexes.csv', header = None)[0].tolist()
        val_dataset = H5ScSingleCellDataset(dir_list=[data_path], dir_labels=class_list, select_channel=[2,3,4], transform=t, return_id=return_id, index_list=[val_set_indexes])

    print("Generating the dataloaders...")
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=28, 
                                                    drop_last=False)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=128, 
                                                    shuffle=False,
                                                    num_workers=28, 
                                                    drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=128, 
                                                    shuffle=False,
                                                    num_workers=28, 
                                                    drop_last=False)

    print("Dataloaders are initialized.")

    wandb_name = "Xenium_ovarian_cancer_finetune" if finetune else "Xenium_ovarian_cancer_scratch"
    checkpoint_epoch_end = ModelCheckpoint(
            filename=f"{wandb_name}_epoch={{epoch}}",
            every_n_epochs=10,
            save_top_k=-1,
            monitor='train_loss',
            save_on_train_epoch_end=True,
            dirpath=savedir
        )
    wandb.init(project="ViT_MAE_Finetuning", 
                name = wandb_name,
                resume="allow")

    wandb_logger = WandbLogger()    
    trainer = pl.Trainer(max_epochs=1000,
                         devices=[0,1],
                         logger=wandb_logger,
                         accelerator="gpu",
                         strategy="ddp",
                         log_every_n_steps=10, 
                         gradient_clip_val=0.5,
                         val_check_interval=0.15,
                         callbacks=[checkpoint_epoch_end])

    model = TransformerModel(finetune=finetune, in_channels=3, return_id=return_id)
    print("Training the model...")
    trainer.validate(model, val_dataloader)
    trainer.fit(model, 
                train_dataloader, 
                val_dataloader, 
                ckpt_path=Path("/ictstr01/groups/ml01/workspace/alioguz.can/scportrait4i/training_output/Xenium_ovarian_cancer_finetune_epoch=epoch=29-v1.ckpt") ## UPDATE PATHS
                )