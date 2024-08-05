import argparse

import numpy as np
import torch

from config import SEED

# fix random seeds for reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import AVSEDataModule
from model import AVSEModule


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    callbacks = [ModelCheckpoint(monitor="val_loss_epoch", filename="model-{epoch:02d}-{global_step}-{val_loss:.2f}"),
                 EarlyStopping(monitor="val_loss", mode="min", patience=5)]

    datamodule = AVSEDataModule(batch_size=args.batch_size, lips=args.lips)
    model = AVSEModule(val_dataset=datamodule.dev_dataset, lr=args.lr)
    trainer = Trainer(default_root_dir=args.log_dir, callbacks=callbacks, deterministic=args.deterministic,
                      log_every_n_steps=args.log_every_n_steps,
                      fast_dev_run=args.fast_dev_run, devices=args.gpus, accelerator=args.accelerator,
                      precision=args.precision, strategy=args.strategy, max_epochs=args.max_epochs,
                      accumulate_grad_batches=args.accumulate_grad_batches, detect_anomaly=args.detect_anomaly,
                      limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
                      num_sanity_val_steps=args.num_sanity_val_steps,
                      gradient_clip_val=args.gradient_clip_val, sync_batchnorm=True,
                      profiler=args.profiler,
                      )
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--lips", type=str2bool, default=False)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument("--detect_anomaly", type=str2bool, default=False)
    parser.add_argument("--fast_dev_run", type=str2bool, default=False)
    parser.add_argument("--auto_lr_find", type=str2bool, default=False)
    parser.add_argument("--auto_scale_batch_size", type=str2bool, default=False)
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--profiler", type=str, default=None)
    args = parser.parse_args()
    main(args)
