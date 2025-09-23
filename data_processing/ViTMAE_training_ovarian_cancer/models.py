import torch
import pytorch_lightning as pl
from transformers import AutoImageProcessor, ViTImageProcessor, ViTMAEForPreTraining, ViTMAEModel, ViTMAEConfig

class TransformerModel(pl.LightningModule):
    def __init__(self, finetune=True, in_channels=3, return_id=True):
        self.pretrained = finetune
        super().__init__()
        self.return_id = return_id
        if finetune:
            self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", output_hidden_states=True)
            self.model.config.mask_ratio = 0.75
        else:
            config = ViTMAEConfig(output_hidden_states=True, mask_ratio=0.75, num_channels=in_channels)
            self.model = ViTMAEForPreTraining(config)
            self.model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(in_channels, 768, kernel_size=(16, 16), stride=(16, 16))
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.return_id:
            imgs, labels, ids = batch
        else:
            imgs, labels = batch
        out = self(imgs)
        loss = out.loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.return_id:
            imgs, labels, ids = batch
        else:
            imgs, labels = batch
        out = self(imgs)
        loss = out.loss
        self.log("val_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer
    
    def unpatchify(self, x):
        p = self.model.vit.embeddings.patch_embeddings.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs