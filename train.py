import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from omegaconf import OmegaConf
import torch
import numpy as np
from src_dev import dataset, models, modules
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def main(cfg):
    # set up seed
    set_seed(cfg.seed)
    
    # set up the dataset and dataloader
    all_dataset = getattr(dataset, cfg.dataset.name)(**cfg.dataset.args)
    train_size = int(cfg.split_ratio * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # set up the model
    model_dict = torch.nn.ModuleDict(
        {name: getattr(models, model.name)(**model.args)
         for name, model in cfg.models.items()}
    )
    
    # Initialize lightning module
    module = getattr(modules, cfg.modules.name)(model_dict, **cfg.modules.args, cfg=cfg)
    
    # Initialize callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='lidar-{epoch:02d}',
        mode='min',
        save_weights_only=True,
        save_top_k=3,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=1000,
        verbose=True,
        mode='min'
    )
    
    logger = TensorBoardLogger(
        save_dir=cfg.output,
        name='logs'
    )
    
    if cfg.checkpoint_path is not None:    
        state = torch.load(cfg.checkpoint_path, weights_only=True)
        module.models.load_state_dict(state)
    
    # set up the trainer
    trainer = Trainer(
        max_epochs=cfg.modules.args.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )
    
    # train the model
    trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)    
    torch.save(module.models.state_dict(), os.path.join(cfg.output, "last.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/sample.yaml",
                        type=str, help="Path to the config file")

    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError(f"Config file {args.config} not found")
    config = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(config, args.__dict__)
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.output, exist_ok=True)
    with open(os.path.join(cfg.output, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    main(cfg)