from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
import wandb

class MethaneSegmentationModel(LightningModule):
    rgb_indices = [54, 36, 21]
    
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net', 'criterion'])
        self.net = net
        self.criterion = criterion
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"].float()
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"].float()
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if batch_idx == 0 and isinstance(self.logger.experiment, wandb.wandb_run.Run):
            img_rgb = images[0, self.rgb_indices, :, :].cpu().detach().numpy().transpose(1, 2, 0)
            label_mask = labels[0, 0, :, :].cpu().detach().numpy()
            pred_mask = preds[0, 0, :, :].cpu().detach().numpy()
            self.logger.experiment.log({
                "val_pred": wandb.Image(
                    img_rgb,
                    masks={
                        "predictions": {"mask_data": pred_mask, "class_labels": {0: "bg", 1: "plume"}},
                        "ground_truth": {"mask_data": label_mask, "class_labels": {0: "bg", 1: "plume"}},
                    },
                    caption="Validation Sample Predictions"
                )
            })
            
        return loss

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]