import os
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric

from src.utils import (
    get_dist_info,
    is_master,
    reduce_value,
)
from src.utils.vis_utils import (
    save_depth_as_uint16png_upload,
    save_image,
    merge_into_row,
    batch_save,
    batch_save_kitti,
    padding_kitti,
)
import sys
sys.path.append("Marigold")

import numpy as np
from PIL import Image
from torchvision import transforms
from model import init_marigold, run_marigold


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


class MarigoldModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        metric,
        monitor,
        save_dir,
        base_lr,
        dataset,
        is_warmup=False,
        
    ):
        super().__init__()
        self.model = init_marigold()

        self.is_warmup = is_warmup

        # loss function
        self.metric = metric
        self.best_result = MinMetric()
        self.save_dir = save_dir

        if is_master():
            self.fieldnames = ["epoch"] + list(self.metric.metrics.keys())
            self.val_csv = os.path.join(self.save_dir, "val.csv")
            self.figure_dir = os.path.join(self.save_dir, "val_results")
            os.makedirs(self.figure_dir, exist_ok=True)

    def load_state_dict(self, path):
        pass

    def forward(self, batch):
        # TODO: loop 
        assert len(batch['rgb']) == 1
        
        image = batch['rgb'][0].cpu()
        gt_depth = batch['gt'][0]
        mask = batch['dep'][0] != 0
        
        depth = run_marigold(self.model, image)
        scale, shift = compute_scale_and_shift(depth, gt_depth, mask)
        depth = scale.view(-1, 1, 1) * depth + shift.view(-1, 1, 1)

        return depth

    # def training_step(self, batch: Any, batch_idx: int):
    #     return 0


    def validation_step(self, batch: Any, batch_idx: int):
        gt = batch["gt"]
        pred = self.forward(batch)
        self.metric.evaluate(pred, gt)
        rank, word_size = get_dist_info()
        result = {}
        result["epoch"] = batch_idx * word_size + rank
        for k in self.metric.metrics.keys():
            result[k] = f"{self.metric.metrics[k].item():.5f}"
        

    def validation_epoch_end(self, outputs: List[Any]):
        avg_metric = self.metric.average()
        for key in avg_metric.keys():
            avg_metric[key] = reduce_value(avg_metric[key])
            self.log(f"val/{key}", avg_metric[key], on_step=False, on_epoch=True)
        self.best_result(avg_metric[self.hparams.monitor])
        self.log(
            "val/best_result", self.best_result.compute(), prog_bar=True, on_epoch=True
        )
        self.metric.reset()
        avg_metric["epoch"] = self.current_epoch
        try:
            self.lr = self.optimizers().param_groups[0]["lr"]
        except:
            pass

    def test_step(self, batch: Any, batch_idx: int):
        rank, word_size = get_dist_info()
        pred = self.forward(batch)
        import jhutil; jhutil.jhprint(1111, self.hparams)

        if self.hparams.dataset in ["nyu", "sunrgbd"]:
            gt = batch["gt"]
            self.metric.evaluate(pred, gt)
            result = {}
            result["filename"] = batch_idx * word_size + rank
            for k in self.metric.metrics.keys():
                result[k] = f"{self.metric.metrics[k].item():.5f}"
        str_i = str(batch_idx * word_size + rank)
        path_i = str_i.zfill(10) + ".png"
        path = os.path.join(self.test_out_dir, path_i)
        if self.hparams.dataset == "kitti":
            save_depth_as_uint16png_upload(pred, path)
        else:
            image = merge_into_row(
                batch["rgb"],
                batch["dep"],
                pred,
                gt,
                (pred - gt).abs(),
                self.hparams.dataset,
            )
            save_image(image, path)

    def test_epoch_end(self, outputs: List[Any]):
        if self.hparams.dataset in ["nyu", "sunrgbd"]:
            avg_metric = self.metric.average()
            for key in avg_metric.keys():
                avg_metric[key] = reduce_value(avg_metric[key])
                self.log(
                    f"test/{key}",
                    avg_metric[key],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            self.metric.reset()
            avg_metric["epoch"] = "Test"

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
        optimizer = self.hparams.optimizer(params)
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"val/{self.hparams.monitor}",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # warm_up
        if self.is_warmup:
            if self.trainer.global_step <= self.warmup_iters:
                lr_warm_up = (
                    self.hparams.base_lr * self.trainer.global_step / self.warmup_iters
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_warm_up
