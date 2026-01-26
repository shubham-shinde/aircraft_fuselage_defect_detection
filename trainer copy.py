
import math
from collections import OrderedDict
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import torchvision
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model
import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    OBB26,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    Pose26,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    Segment26,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2ELoss,
    PoseLoss26,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)
from ultralytics.nn.tasks import BaseModel, parse_model, yaml_model_load

class DetectionModel(BaseModel):
    """YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                output = self.forward(x)
                if self.end2end:
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride, e.g., RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    @property
    def end2end(self):
        """Return whether the model uses end-to-end NMS-free detection."""
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self.model[-1].end2end = value

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails.

        Args:
            y (list[torch.Tensor]): List of detection tensors.

        Returns:
            (list[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class SemiDetectionTrainer(BaseTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)
        
        # CAPLA (Class-Aware Adaptive Pseudolabel Assignment) parameters
        self.capla_reliable_threshold = 0.3  # δ% = 30% for reliable pseudolabels
        self.capla_update_interval = 2000  # Update thresholds every K=2000 iterations
        self.class_adaptive_thresholds = None  # Will be initialized in _setup_train
        self.pseudolabel_scores_per_class = None  # Track scores per class for threshold calculation
        self.iteration_count = 0  # Global iteration counter for periodic updates
        self.total_labeled_samples = 0  # N^l in formula
        self.total_unlabeled_samples = 0  # N^u in formula

    def calculate_class_adaptive_thresholds(self):
        """Calculate class-specific confidence thresholds using CAPLA formula.
        
        Formula: t_c = P_c^l * δ% * n_c^l * (N^u / N^l)
        where:
            P_c^l: sorted list of pseudolabel scores for class c
            δ%: reliable pseudolabel proportion (30%)
            n_c^l: number of ground truth labels for class c
            N^u: total unlabeled data
            N^l: total labeled data
        
        Returns:
            dict: Class-specific adaptive thresholds
        """
        thresholds = {}
        num_classes = self.model.nc
        
        # Count GT labels per class
        class_counts = np.zeros(num_classes)
        for batch in self.train_loader:
            if 'cls' in batch:
                cls = batch['cls'].cpu().numpy().flatten()
                for c in cls:
                    if 0 <= int(c) < num_classes:
                        class_counts[int(c)] += 1
        
        # Calculate ratio of unlabeled to labeled data
        if self.total_labeled_samples > 0:
            data_ratio = self.total_unlabeled_samples / self.total_labeled_samples
        else:
            data_ratio = 1.0
        
        # Calculate threshold for each class
        for class_id in range(num_classes):
            n_c_l = class_counts[class_id]
            
            # Get sorted pseudolabel scores for this class if available
            if (self.pseudolabel_scores_per_class is not None and 
                class_id in self.pseudolabel_scores_per_class and 
                len(self.pseudolabel_scores_per_class[class_id]) > 0):
                
                scores = np.array(self.pseudolabel_scores_per_class[class_id])
                scores.sort()
                
                # Calculate the index for δ% of scores
                idx = int(len(scores) * self.capla_reliable_threshold)
                idx = max(0, min(idx, len(scores) - 1))
                P_c_l = scores[idx]
            else:
                # Default threshold if no scores available
                P_c_l = 0.5
            
            # Apply CAPLA formula
            t_c = P_c_l * self.capla_reliable_threshold * max(n_c_l, 1) * data_ratio
            # Clip threshold between 0 and 1
            t_c = np.clip(t_c, 0.3, 0.95)
            
            thresholds[class_id] = float(t_c)
        
        return thresholds

    def categorize_pseudolabels(self, scores, classes):
        """Categorize pseudolabels as reliable or uncertain based on adaptive thresholds.
        
        Args:
            scores (torch.Tensor): Confidence scores of predictions
            classes (torch.Tensor): Class predictions
            
        Returns:
            tuple: (reliable_mask, uncertain_mask, reliable_indices, uncertain_indices)
        """
        if self.class_adaptive_thresholds is None:
            # Fallback to fixed threshold if adaptive thresholds not yet calculated
            threshold = 0.7
            reliable_mask = scores > threshold
            uncertain_mask = (scores > 0.5) & (scores <= threshold)
        else:
            reliable_mask = torch.zeros_like(scores, dtype=torch.bool)
            uncertain_mask = torch.zeros_like(scores, dtype=torch.bool)
            
            for class_id in range(self.model.nc):
                class_mask = classes == class_id
                if class_mask.sum() > 0:
                    threshold = self.class_adaptive_thresholds.get(class_id, 0.7)
                    reliable_mask[class_mask] = scores[class_mask] > threshold
                    uncertain_mask[class_mask] = (scores[class_mask] > 0.5) & (scores[class_mask] <= threshold)
        
        reliable_indices = torch.where(reliable_mask)[0]
        uncertain_indices = torch.where(uncertain_mask)[0]
        
        return reliable_mask, uncertain_mask, reliable_indices, uncertain_indices


    def combine_labeled_unlabeled(self, batch, batch_unlabel, teacher_pred):
        """Combine labeled and unlabeled batches using CAPLA for semi-supervised learning.

        Args:
            batch (dict): Labeled batch containing images and labels.
            batch_unlabel (dict): Unlabeled batch containing images.
            teacher_pred (torch.Tensor): Predictions from the teacher model on the unlabeled batch.

        Returns:
            dict: Combined batch with labeled and pseudo-labeled data (reliable and uncertain).
        """
        nms_threshold = 0.5
        
        # Initialize tracking for pseudolabel scores per class
        if self.pseudolabel_scores_per_class is None:
            self.pseudolabel_scores_per_class = {i: [] for i in range(self.model.nc)}
        
        # Track reliable and uncertain pseudolabels separately
        reliable_bboxes = []
        reliable_classes = []
        reliable_scores = []
        uncertain_bboxes = []
        uncertain_classes = []
        uncertain_scores = []
        batch_indices_reliable = []
        batch_indices_uncertain = []
        
        # Generate pseudo-labels from teacher predictions using CAPLA
        for i, preds in enumerate(teacher_pred):
            if len(preds) == 0:
                continue
                
            boxes = preds[:, :4]
            scores = preds[:, 4]
            classes = preds[:, 5]
            
            # Categorize predictions as reliable or uncertain using adaptive thresholds
            reliable_mask, uncertain_mask, reliable_idx, uncertain_idx = self.categorize_pseudolabels(
                scores, classes.long()
            )
            
            # Track all scores for threshold recalculation
            for class_id in range(self.model.nc):
                class_scores = scores[classes == class_id].cpu().detach().numpy()
                self.pseudolabel_scores_per_class[class_id].extend(class_scores.tolist())
            
            # Process reliable pseudolabels
            if reliable_idx.shape[0] > 0:
                reliable_boxes = boxes[reliable_idx]
                reliable_scores_sel = scores[reliable_idx]
                reliable_classes_sel = classes[reliable_idx]
                
                # Perform NMS on reliable predictions
                keep_indices_rel = torchvision.ops.nms(reliable_boxes, reliable_scores_sel, nms_threshold)
                reliable_boxes = reliable_boxes[keep_indices_rel]
                reliable_scores_sel = reliable_scores_sel[keep_indices_rel]
                reliable_classes_sel = reliable_classes_sel[keep_indices_rel]
                
                # Convert xyxy to xywh and normalize
                boxes_xywh = torch.zeros_like(reliable_boxes)
                boxes_xywh[:, 0] = (reliable_boxes[:, 0] + reliable_boxes[:, 2]) / 2  # x center
                boxes_xywh[:, 1] = (reliable_boxes[:, 1] + reliable_boxes[:, 3]) / 2  # y center
                boxes_xywh[:, 2] = reliable_boxes[:, 2] - reliable_boxes[:, 0]  # width
                boxes_xywh[:, 3] = reliable_boxes[:, 3] - reliable_boxes[:, 1]  # height
                boxes_xywh = boxes_xywh / batch_unlabel['img'].shape[2]  # normalize
                
                reliable_bboxes.append(boxes_xywh)
                reliable_classes.append(reliable_classes_sel.reshape(-1, 1))
                reliable_scores.append(reliable_scores_sel.reshape(-1, 1))
                batch_indices_reliable.extend([i] * len(reliable_boxes))
            
            # Process uncertain pseudolabels (use with caution in loss)
            if uncertain_idx.shape[0] > 0:
                uncertain_boxes = boxes[uncertain_idx]
                uncertain_scores_sel = scores[uncertain_idx]
                uncertain_classes_sel = classes[uncertain_idx]
                
                # Perform NMS on uncertain predictions
                keep_indices_unc = torchvision.ops.nms(uncertain_boxes, uncertain_scores_sel, nms_threshold)
                uncertain_boxes = uncertain_boxes[keep_indices_unc]
                uncertain_scores_sel = uncertain_scores_sel[keep_indices_unc]
                uncertain_classes_sel = uncertain_classes_sel[keep_indices_unc]
                
                # Convert xyxy to xywh and normalize
                boxes_xywh = torch.zeros_like(uncertain_boxes)
                boxes_xywh[:, 0] = (uncertain_boxes[:, 0] + uncertain_boxes[:, 2]) / 2  # x center
                boxes_xywh[:, 1] = (uncertain_boxes[:, 1] + uncertain_boxes[:, 3]) / 2  # y center
                boxes_xywh[:, 2] = uncertain_boxes[:, 2] - uncertain_boxes[:, 0]  # width
                boxes_xywh[:, 3] = uncertain_boxes[:, 3] - uncertain_boxes[:, 1]  # height
                boxes_xywh = boxes_xywh / batch_unlabel['img'].shape[2]  # normalize
                
                uncertain_bboxes.append(boxes_xywh)
                uncertain_classes.append(uncertain_classes_sel.reshape(-1, 1))
                uncertain_scores.append(uncertain_scores_sel.reshape(-1, 1))
                batch_indices_uncertain.extend([i] * len(uncertain_boxes))
        
        # Concatenate reliable pseudolabels
        if reliable_bboxes:
            reliable_bboxes_cat = torch.cat(reliable_bboxes, dim=0)
            reliable_classes_cat = torch.cat(reliable_classes, dim=0)
            reliable_scores_cat = torch.cat(reliable_scores, dim=0)
            reliable_batch_idx = torch.tensor(batch_indices_reliable, device=reliable_bboxes_cat.device)
            
            batch['batch_idx'] = torch.cat((batch['batch_idx'], reliable_batch_idx + batch['img'].shape[0]), dim=0)
            batch['cls'] = torch.cat((batch['cls'], reliable_classes_cat), dim=0)
            batch['bboxes'] = torch.cat((batch['bboxes'], reliable_bboxes_cat), dim=0)
            
            # Store reliable scores for loss weighting
            if 'reliable_scores' not in batch:
                batch['reliable_scores'] = reliable_scores_cat
            else:
                batch['reliable_scores'] = torch.cat((batch['reliable_scores'], reliable_scores_cat), dim=0)
        
        # Concatenate uncertain pseudolabels with reduced weight
        if uncertain_bboxes:
            uncertain_bboxes_cat = torch.cat(uncertain_bboxes, dim=0)
            uncertain_classes_cat = torch.cat(uncertain_classes, dim=0)
            uncertain_scores_cat = torch.cat(uncertain_scores, dim=0)
            uncertain_batch_idx = torch.tensor(batch_indices_uncertain, device=uncertain_bboxes_cat.device)
            
            batch['batch_idx'] = torch.cat((batch['batch_idx'], uncertain_batch_idx + batch['img'].shape[0]), dim=0)
            batch['cls'] = torch.cat((batch['cls'], uncertain_classes_cat), dim=0)
            batch['bboxes'] = torch.cat((batch['bboxes'], uncertain_bboxes_cat), dim=0)
            
            # Store uncertain scores for loss weighting with lower confidence
            if 'uncertain_scores' not in batch:
                batch['uncertain_scores'] = uncertain_scores_cat * 0.5  # Weight down uncertain labels
            else:
                batch['uncertain_scores'] = torch.cat((batch['uncertain_scores'], uncertain_scores_cat * 0.5), dim=0)
        
        # Extend image file paths and metadata
        batch['im_file'].extend(batch_unlabel['im_file'])
        batch['resized_shape'].extend(batch_unlabel['resized_shape'])
        
        return batch

    def apply_capla_loss_weighting(self, loss, batch):
        """Apply different loss weights based on pseudolabel reliability (CAPLA).
        
        Reliable pseudolabels use standard loss, uncertain pseudolabels use reduced weight.
        
        Args:
            loss (torch.Tensor): Initial loss from model
            batch (dict): Batch with reliability information
            
        Returns:
            torch.Tensor: Weighted loss
        """
        if 'reliable_scores' in batch or 'uncertain_scores' in batch:
            # This is a simplified version - in practice, you would need to integrate
            # this with the actual loss calculation in the model's loss function
            # For now, we track the scores for potential future use
            pass
        return loss

    def _do_train(self):
        """Train the model with the specified world size."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nbt = len(self.train_loader_unlabel) # number of batches for unlabelled data
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        data_loader_unlabel_gen = iter(self.train_loader_unlabel)
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                try: 
                    batch_unlabel = next(data_loader_unlabel_gen)
                except StopIteration:
                    data_loader_unlabel_gen = iter(self.train_loader_unlabel)
                    batch_unlabel = next(data_loader_unlabel_gen)
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    batch_unlabel = self.preprocess_batch(batch_unlabel)
                    del batch_unlabel['ori_shape']
                    with torch.no_grad():
                        teacher_pred = self.model_teacher(batch_unlabel["img"])[0]

                    # convert preds into label using CAPLA
                    batch = self.combine_labeled_unlabeled(batch, batch_unlabel, teacher_pred)
                    
                    # Periodically update CAPLA thresholds
                    self.iteration_count += 1
                    if self.iteration_count % self.capla_update_interval == 0:
                        self.class_adaptive_thresholds = self.calculate_class_adaptive_thresholds()
                        if RANK in {-1, 0}:
                            LOGGER.info(f"Updated CAPLA thresholds at iteration {self.iteration_count}: {self.class_adaptive_thresholds}")
                        # Reset score tracking after update
                        self.pseudolabel_scores_per_class = {i: [] for i in range(self.model.nc)}

                    if self.args.compile:
                        # Decouple inference and loss calculations for improved compile performance
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                # Backward
                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break
                
                # Update Teacher Model with EMA
                student_model_dict = {
                    key: value for key, value in self.model.state_dict().items()
                }
                ema_keep_rate = 0.9996
                new_teacher_dict = OrderedDict()
                for key, value in self.model_teacher.state_dict().items():
                    if key in student_model_dict.keys():
                        new_teacher_dict[key] = (
                                student_model_dict[key] * (1 - ema_keep_rate) + value * ema_keep_rate
                        )
                self.model_teacher.load_state_dict(new_teacher_dict)

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def _setup_train(self):
        """Build dataloaders and optimizer on correct rank process."""
        model_cfg = self.model
        ckpt = self.setup_model()

        model_t, teacher_ckpt = load_checkpoint("runs/detect/teacher_base2/weights/best.pt")
        self.model_teacher = model_t.to(self.device)
        self.teacher_ckpt = teacher_ckpt
        # teacher_weights, _ = load_checkpoint(model_cfg)
        # self.model_teacher = self.get_model(cfg=cfg, weights="runs/detect/teacher_base/weights/best.pt", verbose=RANK == -1).to(self.device)  # calls Model(cfg, weights)
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Compile model
        self.model_teacher = attempt_compile(self.model_teacher, device=self.device, mode=self.args.compile)
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader, self.train_loader_unlabel = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader, _ = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        
        # Initialize CAPLA (Class-Aware Adaptive Pseudolabel Assignment)
        self.pseudolabel_scores_per_class = {i: [] for i in range(self.model.nc)}
        self.iteration_count = 0
        
        # Count dataset sizes for CAPLA formula
        self.total_labeled_samples = len(self.train_loader.dataset) if hasattr(self.train_loader.dataset, '__len__') else 1000
        self.total_unlabeled_samples = len(self.train_loader_unlabel.dataset) if hasattr(self.train_loader_unlabel.dataset, '__len__') else 1000
        
        # Calculate initial adaptive thresholds
        self.class_adaptive_thresholds = self.calculate_class_adaptive_thresholds()
        if RANK in {-1, 0}:
            LOGGER.info(f"Initialized CAPLA with adaptive thresholds: {self.class_adaptive_thresholds}")
            LOGGER.info(f"Labeled samples: {self.total_labeled_samples}, Unlabeled samples: {self.total_unlabeled_samples}")
        
        self.run_callbacks("on_pretrain_routine_end")

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        semi_dataset_path="/media/kaizen/T7/Project/final_project/Project//Dataset/Aircraft_Fuselage_DET2023/unlabel_aircraft_fuselage"
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return (build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs), 
                build_yolo_dataset(self.args, semi_dataset_path, batch, self.data, mode='val', rect=mode == "val", stride=gs))

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset, semi_dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return (build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        ),
        build_dataloader(
            semi_dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        ))


    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        self.model_teacher.nc = self.data["nc"]  # attach number of classes to model
        self.model_teacher.names = self.data["names"]  # attach class names to model
        self.model_teacher.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)

args = dict(
    model="yolo26s.pt",
    data="Dataset/Aircraft_Fuselage_DET2023/aircraft_fuselage_yolo/2026-02-02_5-Fold_Cross-val/split_1/split_1_dataset.yaml",
    batch=8,
    epochs=100,
    )
trainer = SemiDetectionTrainer(overrides=args)
trainer.train()