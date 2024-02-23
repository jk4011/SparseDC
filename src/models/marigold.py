import csv
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
from model import init_marigold, run_marigold, run_marigold_repaint
from src.utils.depth_utils import compute_scale_and_shift, get_depth_dbscan
from typing import Union
import wandb
from src.utils.depth_utils import combine_depth_results


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
        repaint=True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = init_marigold(repaint=repaint)
        self.repaint = repaint

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

    def to(self, device: Union[str, torch.device]):
        self.model.to(device)
        super().to(device)
        import jhutil; jhutil.jhprint(2222, self.model.device)

    def load_state_dict(self, path):
        pass

    def forward(self, batch):
        # TODO: loop
        assert len(batch['rgb']) == 1

        image = batch['rgb'][0]
        depth_gt_sparse = batch['dep'][0][0]

        if self.repaint:
            output = run_marigold_repaint(self.model, image, depth_gt_sparse,
                                          additional_data=batch)
            depth_pred = output["depth_pred"][None, :]  # [1, H, W]
        else:
            depth_pred = run_marigold(self.model, image, additional_data=batch)[None, :]  # [1, H, W]
            # scale, shift = compute_scale_and_shift(depth_pred, depth_gt_sparse, depth_gt_sparse != 0)
            # depth_pred = scale * depth_pred + shift
            # depth_pred_scaled = get_depth_dbscan(depth_pred, depth_gt_sparse)
            # depth_pred = depth_pred_scaled[None, :]

        return depth_pred[None, :]  # [1, 1, H, W]

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
        pred = self.forward(batch)
        
        if self.hparams.dataset in ["nyu", "sunrgbd"]:
            gt = batch["gt"]
            self.metric.evaluate(pred, gt)
            
            if batch_idx % 10 == 0:
                image = batch["rgb"]
                depth_gt = batch["gt"].squeeze()
                depth_gt_sparse = batch["dep"].squeeze()
                
                combined_result = combine_depth_results(image, depth_gt, depth_gt_sparse, pred)
                # TODO: solve error
                wandb.log({"test/rgb": wandb.Image(combined_result)}, step=batch_idx)

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
