import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    # Call test instead of fit, passing the checkpoint path
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

if __name__ == "__main__":
    main()