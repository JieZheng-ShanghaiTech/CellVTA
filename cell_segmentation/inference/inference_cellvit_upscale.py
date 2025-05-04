# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a test set
# Without merging WSI
#
# Aim is to calculate metrics as defined for the CoNIC dataset
#


COLOR_DICT = {
    1: [255, 0, 0],
    2: [34, 221, 77],
    3: [35, 92, 236],
    4: [254, 255, 0],
    5: [255, 159, 68],
    6: [0, 0, 255],
}

TYPE_NUCLEI_DICT = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
    6: 'idn'
}

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import argparse
import logging
from base_ml.base_experiment import BaseExperiment

BaseExperiment.seed_run(1232)

import json
import ujson
import uuid
from collections import deque
from pathlib import Path
from typing import List, Union, Tuple

from cell_segmentation.utils.template_geojson import (
    get_template_point,
    get_template_segmentation,
)

import albumentations as A
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import tqdm
from pandarallel import pandarallel
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from shapely import strtree
from shapely.geometry import Polygon, MultiPolygon
from skimage.color import rgba2rgb
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms

from cell_segmentation.datasets.dataset_coordinator import select_dataset
from models.segmentation.cell_segmentation.cellvit import DataclassHVStorage
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
    get_fast_pq,
    remap_label,
    binarize,
)
from cell_segmentation.utils.post_proc_cellvit import calculate_instances
from cell_segmentation.utils.tools import cropping_center, pair_coordinates
from models.segmentation.cell_segmentation.cellvit import (
    CellViT,
    CellViTUNIAdapter
)


from utils.logger import Logger

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

pandarallel.initialize(progress_bar=False, nb_workers=12)

class InferenceCellViTUpscale:
    def __init__(
        self,
        config_path: str,
        outdir: Union[Path, str],
        gpu: int,
        magnification: int = 40,
    ) -> None:
        """Inference for HoverNet

        Args:
            outdir (Union[Path, str]): Output directory
            gpu (int): CUDA GPU id to use
            magnification (int, optional): Dataset magnification. Defaults to 40.
        """
        self.config_path = Path(config_path)
        self.device = f"cuda:{gpu}"
        self.magnification = magnification
        self.outdir = Path(outdir)

        self.outdir.mkdir(exist_ok=True, parents=True)

        self.__instantiate_logger()
        self.__load_run_conf()

        self.__load_dataset_setup(dataset_path=self.run_conf["data"]["dataset_path"])
        self.__load_model(model_path=self.run_conf["model"]["path"])
        self.__load_inference_transforms()
        self.__setup_amp()
        
        self.logger.info(f"Loaded from: {self.config_path}")

        inference_dataset = select_dataset(
            dataset_name=self.run_conf["data"]["dataset"],
            transforms=self.inference_transforms,
            split="test",
            dataset_config=self.run_conf["data"],
        )

        self.inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=self.run_conf["training"]["batch_size"],
            num_workers=12,
            pin_memory=False,
            shuffle=False,
        )

    def __load_run_conf(self) -> None:
        """Load the config.yaml file with the run setup

        Be careful with loading and usage, since original None values in the run configuration are not stored when dumped to yaml file.
        If you want to check if a key is not defined, first check if the key does exists in the dict.
        """
        with open(self.config_path, "r") as run_config_file:
            yaml_config = yaml.safe_load(run_config_file)
            self.run_conf = dict(yaml_config)
        assert("test_folds" in self.run_conf["data"])

    def __load_dataset_setup(self, dataset_path: Union[Path, str]) -> None:
        """Load the configuration of the cell segmentation dataset.

        The dataset must have a dataset_config.yaml file in their dataset path with the following entries:
            * tissue_types: describing the present tissue types with corresponding integer
            * nuclei_types: describing the present nuclei types with corresponding integer

        Args:
            dataset_path (Union[Path, str]): Path to dataset folder
        """
        dataset_config_path = Path(dataset_path) / "dataset_config.yaml"
        with open(dataset_config_path, "r") as dataset_config_file:
            yaml_config = yaml.safe_load(dataset_config_file)
            self.dataset_config = dict(yaml_config)

    def __instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
            log_dir=self.outdir,
            comment="inference",
            use_timestamp=False,
            formatter="%(message)s",
        )
        self.logger = logger.create_logger()

    def __load_model(self, model_path: Union[Path, str]) -> None:
        """Load model and checkpoint and load the state_dict"""
        self.logger.info(f"Loading model: {model_path}")

        model_checkpoint = torch.load(model_path, map_location="cpu")

        # unpack checkpoint
        self.model = self.__get_model(model_type=model_checkpoint["arch"])
        self.logger.info(
            self.model.load_state_dict(model_checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.to(self.device)

    def __get_model(
        self, model_type: str
    ) -> Union[
        CellViT,
        CellViTUNIAdapter
    ]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViTUNIAdapter

        Returns:
            Union[CellViT, CellViTUNIAdapter]: Model
        """
        implemented_models = [
            "CellViT",
            "CellViTUNIAdapter"
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type == "CellViT":
            model_class = CellViT
            model = model_class(
                num_nuclei_classes=self.run_conf["model"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["model"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )

        if model_type == "CellViTUNIAdapter":
            model_class = CellViTUNIAdapter
            model = model_class(
                        num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                        num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                        drop_rate=0,
                        conv_inplane=64, 
                        n_points=4,
                        deform_num_heads=8, 
                        drop_path_rate=0.4,
                        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
                        with_cffn=True,
                        cffn_ratio=0.25, 
                        deform_ratio=0.5, 
                        add_vit_feature=True)
        
        return model
    
    def __load_inference_transforms(self) -> None:
        """Load the inference transformations from the run_configuration"""
        self.logger.info("Loading inference transformations")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.inference_transforms = A.Compose([A.Normalize(mean=mean, std=std)])

    def __setup_amp(self) -> None:
        """Setup automated mixed precision (amp) for inference."""
        self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)


    def run_patch_inference(self, generate_plots: bool = False) -> None:
        """Run Patch inference with given setup

        Args:
            generate_plots (bool, optional): If inference plots should be generated. Defaults to False.
        """
        self.model.eval()

        # setup score tracker
        image_names = []  # image names as str
        binary_dice_scores = []  # binary dice scores per image
        binary_jaccard_scores = []  # binary jaccard scores per image
        pq_scores = []  # pq-scores per image
        dq_scores = []  # dq-scores per image
        sq_scores = []  # sq-scores per image
        cell_type_pq_scores = []  # pq-scores per cell type and image
        cell_type_dq_scores = []  # dq-scores per cell type and image
        cell_type_sq_scores = []  # sq-scores per cell type and image
        tissue_pred = []  # tissue predictions for each image
        tissue_gt = []  # ground truth tissue image class
        tissue_types_inf = []  # string repr of ground truth tissue image class

        paired_all_global = []  # unique matched index pair
        unpaired_true_all_global = (
            []
        )  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all_global = (
            []
        )  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all_global = []  # each index is 1 independent data point
        pred_inst_type_all_global = []  # each index is 1 independent data point

        # for detections scores
        true_idx_offset = 0
        pred_idx_offset = 0

        inference_loop = tqdm.tqdm(
            enumerate(self.inference_dataloader), total=len(self.inference_dataloader)
        )

        with torch.no_grad():
            for batch_idx, batch in inference_loop:
                batch_metrics = self.inference_step(
                    model=self.model, batch=batch, generate_plots=generate_plots
                )
                image_names = image_names + batch_metrics["image_names"]
                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                dq_scores = dq_scores + batch_metrics["dq_scores"]
                sq_scores = sq_scores + batch_metrics["sq_scores"]
                tissue_types_inf = tissue_types_inf + batch_metrics["tissue_types"]
                cell_type_pq_scores = (
                    cell_type_pq_scores + batch_metrics["cell_type_pq_scores"]
                )
                cell_type_dq_scores = (
                    cell_type_dq_scores + batch_metrics["cell_type_dq_scores"]
                )
                cell_type_sq_scores = (
                    cell_type_sq_scores + batch_metrics["cell_type_sq_scores"]
                )
                tissue_pred.append(batch_metrics["tissue_pred"])
                tissue_gt.append(batch_metrics["tissue_gt"])

                # detection scores
                true_idx_offset = (
                    true_idx_offset + true_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                pred_idx_offset = (
                    pred_idx_offset + pred_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                true_inst_type_all_global.append(batch_metrics["true_inst_type_all"])
                pred_inst_type_all_global.append(batch_metrics["pred_inst_type_all"])
                # increment the pairing index statistic
                batch_metrics["paired_all"][:, 0] += true_idx_offset
                batch_metrics["paired_all"][:, 1] += pred_idx_offset
                paired_all_global.append(batch_metrics["paired_all"])

                batch_metrics["unpaired_true_all"] += true_idx_offset
                batch_metrics["unpaired_pred_all"] += pred_idx_offset
                unpaired_true_all_global.append(batch_metrics["unpaired_true_all"])
                unpaired_pred_all_global.append(batch_metrics["unpaired_pred_all"])

        # assemble batches to datasets (global)
        tissue_types_inf = [t.lower() for t in tissue_types_inf]

        paired_all = np.concatenate(paired_all_global, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all_global, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all_global, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all_global, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all_global, axis=0)
        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        dq_scores = np.array(dq_scores)
        sq_scores = np.array(sq_scores)

        tissue_detection_accuracy = accuracy_score(
            y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
        )
        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=paired_true_type,
            paired_pred=paired_pred_type,
            unpaired_true=unpaired_true_type,
            unpaired_pred=unpaired_pred_type,
        )
        dataset_metrics = {
            "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
            "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
            "Tissue-Multiclass-Accuracy": tissue_detection_accuracy,
            "bPQ": float(np.nanmean(pq_scores)),
            "bDQ": float(np.nanmean(dq_scores)),
            "bSQ": float(np.nanmean(sq_scores)),
            "mPQ": float(np.nanmean([np.nanmean(pq) for pq in cell_type_pq_scores])),
            "mDQ": float(np.nanmean([np.nanmean(dq) for dq in cell_type_dq_scores])),
            "mSQ": float(np.nanmean([np.nanmean(sq) for sq in cell_type_sq_scores])),
            "f1_detection": float(f1_d),
            "precision_detection": float(prec_d),
            "recall_detection": float(rec_d),
        }

        # calculate tissue metrics
        tissue_types = self.dataset_config["tissue_types"]
        tissue_metrics = {}
        for tissue in tissue_types.keys():
            tissue = tissue.lower()
            tissue_ids = np.where(np.asarray(tissue_types_inf) == tissue)
            tissue_metrics[f"{tissue}"] = {}
            tissue_metrics[f"{tissue}"]["Dice"] = float(
                np.nanmean(binary_dice_scores[tissue_ids])
            )
            tissue_metrics[f"{tissue}"]["Jaccard"] = float(
                np.nanmean(binary_jaccard_scores[tissue_ids])
            )
            tissue_metrics[f"{tissue}"]["mPQ"] = float(
                np.nanmean(
                    [np.nanmean(pq) for pq in np.array(cell_type_pq_scores)[tissue_ids]]
                )
            )
            tissue_metrics[f"{tissue}"]["bPQ"] = float(
                np.nanmean(pq_scores[tissue_ids])
            )

        # calculate nuclei metrics
        nuclei_types = self.dataset_config["nuclei_types"]
        nuclei_metrics_d = {}
        nuclei_metrics_pq = {}
        nuclei_metrics_dq = {}
        nuclei_metrics_sq = {}
        for nuc_name, nuc_type in nuclei_types.items():
            if nuc_name.lower() == "background":
                continue
            nuclei_metrics_pq[nuc_name] = np.nanmean(
                [pq[nuc_type] for pq in cell_type_pq_scores]
            )
            nuclei_metrics_dq[nuc_name] = np.nanmean(
                [dq[nuc_type] for dq in cell_type_dq_scores]
            )
            nuclei_metrics_sq[nuc_name] = np.nanmean(
                [sq[nuc_type] for sq in cell_type_sq_scores]
            )
            f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
                paired_true_type,
                paired_pred_type,
                unpaired_true_type,
                unpaired_pred_type,
                nuc_type,
            )
            nuclei_metrics_d[nuc_name] = {
                "f1_cell": f1_cell,
                "prec_cell": prec_cell,
                "rec_cell": rec_cell,
            }

        # print final results
        # binary
        self.logger.info(f"{20*'*'} Binary Dataset metrics {20*'*'}")
        [self.logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]
        # tissue -> the PQ values are bPQ values -> what about mBQ?
        self.logger.info(f"{20*'*'} Tissue metrics {20*'*'}")
        flattened_tissue = []
        for key in tissue_metrics:
            flattened_tissue.append(
                [
                    key,
                    tissue_metrics[key]["Dice"],
                    tissue_metrics[key]["Jaccard"],
                    tissue_metrics[key]["mPQ"],
                    tissue_metrics[key]["bPQ"],
                ]
            )
        self.logger.info(
            tabulate(
                flattened_tissue, headers=["Tissue", "Dice", "Jaccard", "mPQ", "bPQ"]
            )
        )
        # nuclei types
        self.logger.info(f"{20*'*'} Nuclei Type Metrics {20*'*'}")
        flattened_nuclei_type = []
        for key in nuclei_metrics_pq:
            flattened_nuclei_type.append(
                [
                    key,
                    nuclei_metrics_dq[key],
                    nuclei_metrics_sq[key],
                    nuclei_metrics_pq[key],
                ]
            )
        self.logger.info(
            tabulate(flattened_nuclei_type, headers=["Nuclei Type", "DQ", "SQ", "PQ"])
        )
        # nuclei detection metrics
        self.logger.info(f"{20*'*'} Nuclei Detection Metrics {20*'*'}")
        flattened_detection = []
        for key in nuclei_metrics_d:
            flattened_detection.append(
                [
                    key,
                    nuclei_metrics_d[key]["prec_cell"],
                    nuclei_metrics_d[key]["rec_cell"],
                    nuclei_metrics_d[key]["f1_cell"],
                ]
            )
        self.logger.info(
            tabulate(
                flattened_detection,
                headers=["Nuclei Type", "Precision", "Recall", "F1"],
            )
        )

        # save all folds
        image_metrics = {}
        for idx, image_name in enumerate(image_names):
            image_metrics[image_name] = {
                "Dice": float(binary_dice_scores[idx]),
                "Jaccard": float(binary_jaccard_scores[idx]),
                "bPQ": float(pq_scores[idx]),
            }
        all_metrics = {
            "dataset": dataset_metrics,
            "tissue_metrics": tissue_metrics,
            "image_metrics": image_metrics,
            "nuclei_metrics_pq": nuclei_metrics_pq,
            "nuclei_metrics_d": nuclei_metrics_d,
        }

        # saving
        with open(str(self.outdir / "inference_results.json"), "w") as outfile:
            json.dump(all_metrics, outfile, indent=2)


    def inference_step(
        self,
        model: Union[
            CellViT,
            CellViTUNIAdapter
        ],
        batch: tuple,
        generate_plots: bool = False,
    ) -> dict:
        """Inference step for a patch-wise batch

        Args:
            model (CellViT, CellViTUNIAdapter): Model to use for inference
            batch (tuple): Batch with the following structure:
                * Images (torch.Tensor)
                * Masks (dict)
                * Tissue types as str
                * Image name as str
            generate_plots (bool, optional):  If inference plots should be generated. Defaults to False.
        """
        # unpack batch, for shape compare train_step method
        imgs = batch[0]
        masks = batch[1]
        tissue_types = list(batch[2])
        image_names = list(batch[3])

        flattened_upscaled_imgs, upscaled_size, position_order = self.upscale_imgs(
            imgs, 
            grid=(2, 2), 
            patch_size=self.run_conf['data']['input_shape'], 
            overlap=self.run_conf['data']['overlap'],
            with_padding=self.run_conf['data']['with_padding']
        )
        flattened_upscaled_imgs = flattened_upscaled_imgs.to(self.device)

        model.zero_grad()
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions_flattened = model.forward(flattened_upscaled_imgs)
        else:
            predictions_flattened = model.forward(flattened_upscaled_imgs)
        
        cell_dict_list = self.extract_unflattened_cell_dict(
            predictions_flattened, 
            position_order=position_order, 
            overlap=self.run_conf['data']['overlap'],
            with_padding=self.run_conf['data']['with_padding'],
            generate_plots = generate_plots
        )
        predictions = self.extract_downsampled_map_from_cell_dict(
            cell_dict_list, 
            upscaled_size=upscaled_size,
            tissue_types = torch.mean(
                predictions_flattened["tissue_types"].unfold(0, len(position_order), len(position_order)), 
                dim=-1
            )
        )
        gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)

        # scores
        batch_metrics, scores = self.calculate_step_metric(predictions, gt, image_names)
        batch_metrics["tissue_types"] = tissue_types
        if generate_plots:
            self.plot_results(
                imgs=imgs,
                predictions=predictions,
                ground_truth=gt,
                img_names=image_names,
                num_nuclei_classes=len(self.dataset_config['nuclei_types']),
                outdir=Path(self.outdir / "inference_predictions"),
                scores=scores,
            )

        return batch_metrics


    def upscale_imgs(self, imgs: torch.Tensor, grid: Tuple = (2, 2), patch_size: int = 256, overlap: int = 32, with_padding: bool = False):
        if with_padding:
            imgs_upscaled = F.interpolate(imgs, size=(grid[0] * (patch_size - overlap), grid[1] * (patch_size - overlap)), mode='bilinear', align_corners=False)
            imgs_upscaled = F.pad(imgs_upscaled, ([overlap//2 for _ in range(4)]), mode='constant', value=1)
        else:
            imgs_upscaled = F.interpolate(imgs, size=(grid[0] * (patch_size - overlap) + overlap, grid[1] * (patch_size - overlap) + overlap), mode='bilinear', align_corners=False)

        split_imgs = imgs_upscaled.unfold(2, patch_size, patch_size-overlap).unfold(3, patch_size, patch_size-overlap)
        flattened_imgs = split_imgs.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size).to(self.device) # 
        position_order = [(i, j) for i in range(grid[0]) for j in range(grid[1])]
        return flattened_imgs, imgs_upscaled.shape[-1], position_order


    def extract_unflattened_cell_dict(self, predictions, position_order, overlap: int = 64, with_padding: bool = False, generate_plots: bool = False):

        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )  # shape: (4*batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (4*batch_size, num_nuclei_classes, H, W)
        _, instance_types = self.model.calculate_instance_map(predictions)

        # unpack each patch from batch
        cell_dict_list = []
        batch_size = len(instance_types) // len(position_order)
        for i in range(batch_size):
            cell_dict_upscaled_patch = []
            for j in range(len(position_order)):
                
                row, col = position_order[j]
                patch_instance_types = instance_types[i * len(position_order) + j]

                # calculate coordinate on highest magnifications
                if with_padding:
                    x_global = int(
                        row * self.run_conf["data"]["input_shape"] - (row + 0.5) * overlap
                    )
                    y_global = int(
                        col * self.run_conf["data"]["input_shape"] - (col + 0.5) * overlap
                    )
                else:
                    x_global = int(
                        row * (self.run_conf["data"]["input_shape"] - overlap)
                    )
                    y_global = int(
                        col * (self.run_conf["data"]["input_shape"] - overlap)
                    )

                # extract cell information
                for cell in patch_instance_types.values():
                    if cell["type"] == self.dataset_config["nuclei_types"]["background"]:
                        continue
                    offset_global = np.array([x_global, y_global])
                    centroid_global = cell["centroid"] + np.flip(offset_global)
                    contour_global = cell["contour"] + np.flip(offset_global)
                    bbox_global = cell["bbox"] + offset_global
                    cell_dict = {
                        "bbox": bbox_global.tolist(),
                        "centroid": centroid_global.tolist(),
                        "contour": contour_global.tolist(),
                        "type_prob": cell["type_prob"],
                        "type": cell["type"],
                        "patch_coordinates": [
                            row,
                            col,
                        ],
                        "cell_status": get_cell_position_marging(
                            cell["bbox"], 256, overlap
                        ),
                        "offset_global": offset_global.tolist()
                    }
                    if np.max(cell["bbox"]) == 256 or np.min(cell["bbox"]) == 0:
                        position = get_cell_position(cell["bbox"], 256)
                        cell_dict["edge_position"] = True
                        cell_dict["edge_information"] = {}
                        cell_dict["edge_information"]["position"] = position
                        cell_dict["edge_information"][
                            "edge_patches"
                        ] = get_edge_patch(
                            position, row, col
                        )
                    else:
                        cell_dict["edge_position"] = False

                    cell_dict_upscaled_patch.append(cell_dict)
            
            cell_dict_upscaled_patch = self.post_process_edge_cells(cell_dict_upscaled_patch)
            # self.logger.info("Converting segmentation to geojson")
            geojson_list = convert_geojson(cell_dict_upscaled_patch, True)
            if generate_plots:
                with open(os.path.join(self.outdir, "cells.geojson"), "w") as outfile:
                    ujson.dump(geojson_list, outfile, indent=2)
            cell_dict_list.append(cell_dict_upscaled_patch)

        return cell_dict_list


    def extract_downsampled_map_from_cell_dict(self, cell_dict_list: List[pd.DataFrame], upscaled_size: int, tissue_types) :
        org_size = self.run_conf["data"]["input_shape"]
        masks_list = []
        for cell_dict in cell_dict_list:
            inst_map = np.zeros((org_size, org_size), dtype=np.int32)
            type_map = np.zeros((org_size, org_size), dtype=np.int32)
            for ind, (_, cell) in enumerate(cell_dict.iterrows()):
                cv2.fillPoly(inst_map, [(np.array(cell['contour']) * org_size / upscaled_size).astype(np.int32)], ind)
                cv2.fillPoly(type_map, [(np.array(cell['contour']) * org_size / upscaled_size).astype(np.int32)], cell['type'])
            mask = np.stack([inst_map, type_map], axis=-1)
            inst_map = mask[:, :, 0].copy()
            type_map = mask[:, :, 1].copy()
            np_map = mask[:, :, 0].copy()
            np_map[np_map > 0] = 1
            hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)

            masks = {
                "instance_map": torch.Tensor(inst_map).type(torch.int64),
                "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
                "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
                "hv_map": torch.Tensor(hv_map).type(torch.float32),
            }
            masks_list.append(masks)

        for keys in masks.keys():
            masks[keys] = torch.stack([masks[keys] for masks in masks_list], dim=0)
            
        predictions = self.unpack_predictions(masks=masks, tissue_types=tissue_types)

        return predictions


    def post_process_edge_cells(self, cell_list: List[dict]) -> pd.DataFrame:
        """Use the CellPostProcessor to remove multiple cells and merge due to overlap

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global

        Returns:
            cleaned_cells (pd.DataFrame): Dataframe with the post-processed cells
        """

        cell_processor = CellPostProcessor(cell_list, self.logger)
        cleaned_cells = cell_processor.post_process_cells()

        return cleaned_cells


    def calculate_step_metric(
        self,
        predictions: DataclassHVStorage,
        gt: DataclassHVStorage,
        image_names: list[str],
    ) -> Tuple[dict, list]:
        """Calculate the metrics for the validation step

        Args:
            predictions (DataclassHVStorage): Processed network output
            gt (DataclassHVStorage): Ground truth values
            image_names (list(str)): List with image names

        Returns:
            Tuple[dict, list]:
                * dict: Dictionary with metrics. Structure not fixed yet
                * list with cell_dice, cell_jaccard and pq for each image
        """
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        # preparation and device movement
        predictions["tissue_types_classes"] = F.softmax(
            predictions["tissue_types"], dim=-1
        )
        pred_tissue = (
            torch.argmax(predictions["tissue_types_classes"], dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        # segmentation scores
        binary_dice_scores = []  # binary dice scores per image
        binary_jaccard_scores = []  # binary jaccard scores per image
        pq_scores = []  # pq-scores per image
        dq_scores = []  # dq-scores per image
        sq_scores = []  # sq_scores per image
        cell_type_pq_scores = []  # pq-scores per cell type and image
        cell_type_dq_scores = []  # dq-scores per cell type and image
        cell_type_sq_scores = []  # sq-scores per cell type and image
        scores = []  # all scores in one list

        # detection scores
        paired_all = []  # unique matched index pair
        unpaired_true_all = (
            []
        )  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all = (
            []
        )  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all = []  # each index is 1 independent data point
        pred_inst_type_all = []  # each index is 1 independent data point

        # for detections scores
        true_idx_offset = 0
        pred_idx_offset = 0

        for i in range(len(pred_tissue)):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

            # pq values
            if len(np.unique(instance_maps_gt[i])) == 1:
                dq, sq, pq = np.nan, np.nan, np.nan
            else:
                remapped_instance_pred = binarize(
                    predictions["instance_types_nuclei"][i][1:].transpose(1, 2, 0)
                )
                remapped_gt = remap_label(instance_maps_gt[i])
                [dq, sq, pq], _ = get_fast_pq(
                    true=remapped_gt, pred=remapped_instance_pred
                )
            pq_scores.append(pq)
            dq_scores.append(dq)
            sq_scores.append(sq)
            scores.append(
                [
                    cell_dice.detach().cpu().numpy(),
                    cell_jaccard.detach().cpu().numpy(),
                    pq,
                ]
            )

            # pq values per class (with class 0 beeing background -> should be skipped in the future)
            nuclei_type_pq = []
            nuclei_type_dq = []
            nuclei_type_sq = []
            for j in range(0, len(self.dataset_config['nuclei_types'])):
                pred_nuclei_instance_class = remap_label(
                    predictions["instance_types_nuclei"][i][j, ...]
                )
                target_nuclei_instance_class = remap_label(
                    gt["instance_types_nuclei"][i][j, ...]
                )

                # if ground truth is empty, skip from calculation
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                    dq_tmp = np.nan
                    sq_tmp = np.nan
                else:
                    [dq_tmp, sq_tmp, pq_tmp], _ = get_fast_pq(
                        pred_nuclei_instance_class,
                        target_nuclei_instance_class,
                        match_iou=0.5,
                    )
                nuclei_type_pq.append(pq_tmp)
                nuclei_type_dq.append(dq_tmp)
                nuclei_type_sq.append(sq_tmp)

            # detection scores
            true_centroids = np.array(
                [v["centroid"] for k, v in gt["instance_types"][i].items()]
            )
            true_instance_type = np.array(
                [v["type"] for k, v in gt["instance_types"][i].items()]
            )
            pred_centroids = np.array(
                [v["centroid"] for k, v in predictions["instance_types"][i].items()]
            )
            pred_instance_type = np.array(
                [v["type"] for k, v in predictions["instance_types"][i].items()]
            )

            if true_centroids.shape[0] == 0:
                true_centroids = np.array([[0, 0]])
                true_instance_type = np.array([0])
            if pred_centroids.shape[0] == 0:
                pred_centroids = np.array([[0, 0]])
                pred_instance_type = np.array([0])
            if self.magnification == 40:
                pairing_radius = 12
            else:
                pairing_radius = 6
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroids, pred_centroids, pairing_radius
            )
            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
            )
            true_inst_type_all.append(true_instance_type)
            pred_inst_type_all.append(pred_instance_type)

            # increment the pairing index statistic
            if paired.shape[0] != 0:  # ! sanity
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

            cell_type_pq_scores.append(nuclei_type_pq)
            cell_type_dq_scores.append(nuclei_type_dq)
            cell_type_sq_scores.append(nuclei_type_sq)

        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        batch_metrics = {
            "image_names": image_names,
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "cell_type_pq_scores": cell_type_pq_scores,
            "cell_type_dq_scores": cell_type_dq_scores,
            "cell_type_sq_scores": cell_type_sq_scores,
            "tissue_pred": pred_tissue,
            "tissue_gt": gt["tissue_types"],
            "paired_all": paired_all,
            "unpaired_true_all": unpaired_true_all,
            "unpaired_pred_all": unpaired_pred_all,
            "true_inst_type_all": true_inst_type_all,
            "pred_inst_type_all": pred_inst_type_all,
        }

        return batch_metrics, scores


    def unpack_predictions(
        self, masks: dict, tissue_types: list
    ) -> DataclassHVStorage:
        # get ground truth values, perform one hot encoding for segmentation maps
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        ).type(
            torch.float32
        )  # background, nuclei
        nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=len(self.dataset_config['nuclei_types'])
        ).type(
            torch.float32
        )  # background + nuclei types

        # assemble ground truth dictionary
        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, num_nuclei_classes)
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, 2)
            "hv_map": masks["hv_map"].to(self.device),  # shape: (batch_size, H, W, 2)
            "instance_map": masks["instance_map"].to(
                self.device
            ),  # shape: (batch_size, H, W) -> each instance has one integer
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
            "tissue_types": tissue_types.to(self.device),  # shape: batch_size
        }
        gt["instance_types"] = calculate_instances(
            gt["nuclei_type_map"], gt["instance_map"]
        )
        gt = DataclassHVStorage(**gt, batch_size=gt["tissue_types"].shape[0])
        return gt


    def unpack_masks(
        self, masks: dict, tissue_types: list
    ) -> DataclassHVStorage:
        # get ground truth values, perform one hot encoding for segmentation maps
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        ).type(
            torch.float32
        )  # background, nuclei
        nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=len(self.dataset_config['nuclei_types'])
        ).type(
            torch.float32
        )  # background + nuclei types

        # assemble ground truth dictionary
        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, num_nuclei_classes)
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, 2)
            "hv_map": masks["hv_map"].to(self.device),  # shape: (batch_size, H, W, 2)
            "instance_map": masks["instance_map"].to(
                self.device
            ),  # shape: (batch_size, H, W) -> each instance has one integer
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
            "tissue_types": torch.Tensor(
                [self.dataset_config["tissue_types"][t] for t in tissue_types]
            )
            .type(torch.LongTensor)
            .to(self.device),  # shape: batch_size
        }
        gt["instance_types"] = calculate_instances(
            gt["nuclei_type_map"], gt["instance_map"]
        )
        gt = DataclassHVStorage(**gt, batch_size=gt["tissue_types"].shape[0])
        return gt


    def plot_results(
        self,
        imgs: Union[torch.Tensor, np.ndarray],
        predictions: dict,
        ground_truth: dict,
        img_names: List,
        num_nuclei_classes: int,
        outdir: Union[Path, str],
        scores: List[List[float]] = None,
    ) -> None:
        # TODO: Adapt Docstring and function, currently not working with our shape
        """Generate example plot with image, binary_pred, hv-map and instance map from prediction and ground-truth

        Args:
            imgs (Union[torch.Tensor, np.ndarray]): Images to process, a random number (num_images) is selected from this stack
                Shape: (batch_size, 3, H', W')
            predictions (dict): Predictions of models. Keys:
                "nuclei_type_map": Shape: (batch_size, H', W', num_nuclei)
                "nuclei_binary_map": Shape: (batch_size, H', W', 2)
                "hv_map": Shape: (batch_size, H', W', 2)
                "instance_map": Shape: (batch_size, H', W')
            ground_truth (dict): Ground truth values. Keys:
                "nuclei_type_map": Shape: (batch_size, H', W', num_nuclei)
                "nuclei_binary_map": Shape: (batch_size, H', W', 2)
                "hv_map": Shape: (batch_size, H', W', 2)
                "instance_map": Shape: (batch_size, H', W')
            img_names (List): Names of images as list
            num_nuclei_classes (int): Number of total nuclei classes including background
            outdir (Union[Path, str]): Output directory where images should be stored
            scores (List[List[float]], optional): List with scores for each image.
                Each list entry is a list with 3 scores: Dice, Jaccard and bPQ for the image.
                Defaults to None.
        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        h = ground_truth.hv_map.shape[2]
        w = ground_truth.hv_map.shape[3]

        # convert to rgb and crop to selection
        sample_images = (
            imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )  # convert to rgb
        sample_images = cropping_center(sample_images, (h, w), True)

        pred_sample_binary_map = (
            predictions.nuclei_binary_map[:, 1, :, :].detach().cpu().numpy()
        )
        pred_sample_hv_map = predictions.hv_map.detach().cpu().numpy()
        pred_sample_instance_maps = predictions.instance_map.detach().cpu().numpy()
        pred_sample_type_maps = (
            torch.argmax(predictions.nuclei_type_map, dim=1).detach().cpu().numpy()
        )

        gt_sample_binary_map = ground_truth.nuclei_binary_map.detach().cpu().numpy()
        gt_sample_hv_map = ground_truth.hv_map.detach().cpu().numpy()
        gt_sample_instance_map = ground_truth.instance_map.detach().cpu().numpy()
        gt_sample_type_map = (
            torch.argmax(ground_truth.nuclei_type_map, dim=1).detach().cpu().numpy()
        )
        # gt_sample_binary_map = (
        #     torch.argmax(ground_truth["nuclei_binary_map"], dim=-1).detach().cpu()
        # )

        # create colormaps
        hv_cmap = plt.get_cmap("jet")
        binary_cmap = plt.get_cmap("jet")
        instance_map = plt.get_cmap("viridis")
        cell_colors = ["#ffffff", "#ff0000", "#00ff00", "#1e00ff", "#feff00", "#ffbf00", "#ff00bf"]

        # invert the normalization of the sample images
        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        inv_normalize = transforms.Normalize(
            mean=[-0.5 / mean[0], -0.5 / mean[1], -0.5 / mean[2]],
            std=[1 / std[0], 1 / std[1], 1 / std[2]],
        )
        inv_samples = inv_normalize(torch.tensor(sample_images).permute(0, 3, 1, 2))
        sample_images = inv_samples.permute(0, 2, 3, 1).detach().cpu().numpy()

        for i in range(len(img_names)):
            fig, axs = plt.subplots(figsize=(6, 2), dpi=300)
            placeholder = np.zeros((2 * h, 7 * w, 3))
            # orig image
            placeholder[:h, :w, :3] = sample_images[i]
            placeholder[h : 2 * h, :w, :3] = sample_images[i]
            # binary prediction
            placeholder[:h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_binary_map[i] * 255)
            )
            placeholder[h : 2 * h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_binary_map[i] * 255)
            )  
            # hv maps
            placeholder[:h, 2 * w : 3 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, 0, :, :] + 1) / 2)
            )
            placeholder[h : 2 * h, 2 * w : 3 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, 0, :, :] + 1) / 2)
            )
            placeholder[:h, 3 * w : 4 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, 1, :, :] + 1) / 2)
            )
            placeholder[h : 2 * h, 3 * w : 4 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, 1, :, :] + 1) / 2)
            )
            # instance_predictions
            placeholder[:h, 4 * w : 5 * w, :3] = rgba2rgb(
                instance_map(
                    (gt_sample_instance_map[i] - np.min(gt_sample_instance_map[i]))
                    / (
                        np.max(gt_sample_instance_map[i])
                        - np.min(gt_sample_instance_map[i] + 1e-10)
                    )
                )
            )
            placeholder[h : 2 * h, 4 * w : 5 * w, :3] = rgba2rgb(
                instance_map(
                    (
                        pred_sample_instance_maps[i]
                        - np.min(pred_sample_instance_maps[i])
                    )
                    / (
                        np.max(pred_sample_instance_maps[i])
                        - np.min(pred_sample_instance_maps[i] + 1e-10)
                    )
                )
            )
            # type_predictions
            placeholder[:h, 5 * w : 6 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_type_map[i] / num_nuclei_classes)
            )
            placeholder[h : 2 * h, 5 * w : 6 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_type_maps[i] / num_nuclei_classes)
            )

            # contours
            # gt
            gt_contours_polygon = [
                v["contour"] for v in ground_truth.instance_types[i].values()
            ]
            gt_contours_polygon = [
                list(zip(poly[:, 0], poly[:, 1])) for poly in gt_contours_polygon
            ]
            gt_contour_colors_polygon = [
                cell_colors[v["type"]]
                for v in ground_truth.instance_types[i].values()
            ]
            gt_cell_image = Image.fromarray(
                (sample_images[i] * 255).astype(np.uint8)
            ).convert("RGB")
            gt_drawing = ImageDraw.Draw(gt_cell_image)
            add_patch = lambda poly, color: gt_drawing.polygon(
                poly, outline=color, width=2
            )
            [
                add_patch(poly, c)
                for poly, c in zip(gt_contours_polygon, gt_contour_colors_polygon)
            ]
            gt_cell_image.save(outdir / f"raw_gt_{img_names[i]}")
            placeholder[:h, 6 * w : 7 * w, :3] = np.asarray(gt_cell_image) / 255
            # pred
            pred_contours_polygon = [
                v["contour"] for v in predictions.instance_types[i].values()
            ]
            pred_contours_polygon = [
                list(zip(poly[:, 0], poly[:, 1])) for poly in pred_contours_polygon
            ]
            pred_contour_colors_polygon = [
                cell_colors[v["type"]]
                for v in predictions.instance_types[i].values()
            ]
            pred_cell_image = Image.fromarray(
                (sample_images[i] * 255).astype(np.uint8)
            ).convert("RGB")
            pred_drawing = ImageDraw.Draw(pred_cell_image)
            add_patch = lambda poly, color: pred_drawing.polygon(
                poly, outline=color, width=2
            )
            [
                add_patch(poly, c)
                for poly, c in zip(pred_contours_polygon, pred_contour_colors_polygon)
            ]
            pred_cell_image.save(outdir / f"raw_pred_{img_names[i]}")
            placeholder[h : 2 * h, 6 * w : 7 * w, :3] = (
                np.asarray(pred_cell_image) / 255
            )

            # plotting
            axs.imshow(placeholder)
            axs.set_xticks(np.arange(w / 2, 7 * w, w))
            axs.set_xticklabels(
                [
                    "Image",
                    "Binary-Cells",
                    "HV-Map-0",
                    "HV-Map-1",
                    "Instances",
                    "Nuclei-Pred",
                    "Countours",
                ],
                fontsize=6,
            )
            axs.xaxis.tick_top()

            axs.set_yticks(np.arange(h / 2, 2 * h, h))
            axs.set_yticklabels(["GT", "Pred."], fontsize=6)
            axs.tick_params(axis="both", which="both", length=0)
            grid_x = np.arange(w, 6 * w, w)
            grid_y = np.arange(h, 2 * h, h)

            for x_seg in grid_x:
                axs.axvline(x_seg, color="black")
            for y_seg in grid_y:
                axs.axhline(y_seg, color="black")

            if scores is not None:
                axs.text(
                    20,
                    1.85 * h,
                    f"Dice: {str(np.round(scores[i][0], 2))}\nJac.: {str(np.round(scores[i][1], 2))}\nbPQ: {str(np.round(scores[i][2], 2))}",
                    bbox={"facecolor": "white", "pad": 2, "alpha": 0.5},
                    fontsize=4,
                )
            fig.suptitle(f"Patch Predictions for {img_names[i]}")
            fig.tight_layout()
            fig.savefig(outdir / f"pred_{img_names[i]}")
            plt.close()


class CellPostProcessor:
    def __init__(self, cell_list: List[dict], logger: logging.Logger) -> None:
        """POst-Processing a list of cells from one WSI

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global
            logger (logging.Logger): Logger
        """
        self.logger = logger
        # self.logger.info("Initializing Cell-Postprocessor")
        try:
            self.cell_df = pd.DataFrame(cell_list)
            self.cell_df = self.cell_df.parallel_apply(convert_coordinates, axis=1)
        except:
            self.cell_df = pd.DataFrame(cell_list, columns=['bbox', 'centroid', 'contour', 'type_prob', 'type', 'patch_coordinates',
                                        'cell_status', 'offset_global', 'edge_position', 'edge_information'])

        self.mid_cells = self.cell_df[
            self.cell_df["cell_status"] == 0
        ]  # cells in the mid
        self.cell_df_margin = self.cell_df[
            self.cell_df["cell_status"] != 0
        ]  # cells either torching the border or margin

    def post_process_cells(self) -> pd.DataFrame:
        """Main Post-Processing coordinator, entry point

        Returns:
            pd.DataFrame: DataFrame with post-processed and cleaned cells
        """
        # self.logger.info("Finding edge-cells for merging")
        cleaned_edge_cells = self._clean_edge_cells()
        # self.logger.info("Removal of cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = pd.concat(
            [self.mid_cells, cleaned_edge_cells]
        ).sort_index()
        return postprocessed_cells

    def _clean_edge_cells(self) -> pd.DataFrame:
        """Create a DataFrame that just contains all margin cells (cells inside the margin, not touching the border)
        and border/edge cells (touching border) with no overlapping equivalent (e.g, if patch has no neighbour)

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """

        margin_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 0
        ]  # cells at the margin, but not touching the border
        edge_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 1
        ]  # cells touching the border
        existing_patches = list(set(self.cell_df_margin["patch_coordinates"].to_list()))

        edge_cells_unique = pd.DataFrame(
            columns=self.cell_df_margin.columns
        )  # cells torching the border without having an overlap from other patches
        for idx, cell_info in edge_cells.iterrows():
            edge_information = dict(cell_info["edge_information"])
            edge_patch = edge_information["edge_patches"][0]
            edge_patch = f"{edge_patch[0]}_{edge_patch[1]}"
            if edge_patch not in existing_patches:
                edge_cells_unique.loc[idx, :] = cell_info

        cleaned_edge_cells = pd.concat([margin_cells, edge_cells_unique])

        return cleaned_edge_cells.sort_index()

    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping cells from provided DataFrame

        Args:
            cleaned_edge_cells (pd.DataFrame): DataFrame that should be cleaned

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        merged_cells = cleaned_edge_cells

        for iteration in range(20):
            poly_list = []
            for idx, cell_info in merged_cells.iterrows():
                poly = Polygon(cell_info["contour"])
                if not poly.is_valid:
                    self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                    multi = poly.buffer(0)
                    if isinstance(multi, MultiPolygon):
                        if len(multi) > 1:
                            poly_idx = np.argmax([p.area for p in multi])
                            poly = multi[poly_idx]
                            poly = Polygon(poly)
                        else:
                            poly = multi[0]
                            poly = Polygon(poly)
                    else:
                        poly = Polygon(multi)
                poly.uid = idx
                poly_list.append(poly)

            # use an strtree for fast querying
            tree = strtree.STRtree(poly_list)

            merged_idx = deque()
            iterated_cells = set()
            overlaps = 0

            for query_poly in poly_list:
                if query_poly.uid not in iterated_cells:
                    intersected_polygons = tree.query(
                        query_poly
                    )  # this also contains a self-intersection
                    if (
                        len(intersected_polygons) > 1
                    ):  # we have more at least one intersection with another cell
                        submergers = []  # all cells that overlap with query
                        for inter_poly in intersected_polygons:
                            if (
                                inter_poly.uid != query_poly.uid
                                and inter_poly.uid not in iterated_cells
                            ):
                                if (
                                    query_poly.intersection(inter_poly).area
                                    / query_poly.area
                                    > 0.01
                                    or query_poly.intersection(inter_poly).area
                                    / inter_poly.area
                                    > 0.01
                                ):
                                    overlaps = overlaps + 1
                                    submergers.append(inter_poly)
                                    iterated_cells.add(inter_poly.uid)
                        # catch block: empty list -> some cells are touching, but not overlapping strongly enough
                        if len(submergers) == 0:
                            merged_idx.append(query_poly.uid)
                        else:  # merging strategy: take the biggest cell, other merging strategies needs to get implemented
                            selected_poly_index = np.argmax(
                                np.array([p.area for p in submergers])
                            )
                            selected_poly_uid = submergers[selected_poly_index].uid
                            merged_idx.append(selected_poly_uid)
                    else:
                        # no intersection, just add
                        merged_idx.append(query_poly.uid)
                    iterated_cells.add(query_poly.uid)

            # self.logger.info(
            #     f"Iteration {iteration}: Found overlap of # cells: {overlaps}"
            # )
            if overlaps == 0:
                # self.logger.info("Found all overlapping cells")
                break
            elif iteration == 20:
                self.logger.info(
                    f"Not all doubled cells removed, still {overlaps} to remove. For perfomance issues, we stop iterations now. Please raise an issue in git or increase number of iterations."
                )
            merged_cells = cleaned_edge_cells.loc[
                cleaned_edge_cells.index.isin(merged_idx)
            ].sort_index()

        return merged_cells.sort_index()


def convert_coordinates(row: pd.Series) -> pd.Series:
    """Convert a row from x,y type to one string representation of the patch position for fast querying
    Repr: x_y

    Args:
        row (pd.Series): Row to be processed

    Returns:
        pd.Series: Processed Row
    """
    x, y = row["patch_coordinates"]
    row["patch_row"] = x
    row["patch_col"] = y
    row["patch_coordinates"] = f"{x}_{y}"
    return row

def get_cell_position(bbox: np.ndarray, patch_size: int = 1024) -> List[int]:
    """Get cell position as a list

    Entry is 1, if cell touches the border: [top, right, down, left]

    Args:
        bbox (np.ndarray): Bounding-Box of cell
        patch_size (int, optional): Patch-size. Defaults to 1024.

    Returns:
        List[int]: List with 4 integers for each position
    """
    # bbox = 2x2 array in h, w style
    # bbox[0,0] = upper position (height)
    # bbox[1,0] = lower dimension (height)
    # boox[0,1] = left position (width)
    # bbox[1,1] = right position (width)
    # bbox[:,0] -> x dimensions
    top, left, down, right = False, False, False, False
    if bbox[0, 0] == 0:
        top = True
    if bbox[0, 1] == 0:
        left = True
    if bbox[1, 0] == patch_size:
        down = True
    if bbox[1, 1] == patch_size:
        right = True
    position = [top, right, down, left]
    position = [int(pos) for pos in position]

    return position

def get_cell_position_marging(
    bbox: np.ndarray, patch_size: int = 1024, margin: int = 64
) -> int:
    """Get the status of the cell, describing the cell position

    A cell is either in the mid (0) or at one of the borders (1-8)

    # Numbers are assigned clockwise, starting from top left
    # i.e., top left = 1, top = 2, top right = 3, right = 4, bottom right = 5 bottom = 6, bottom left = 7, left = 8
    # Mid status is denoted by 0

    Args:
        bbox (np.ndarray): Bounding Box of cell
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        margin (int, optional): Margin-Size. Defaults to 64.

    Returns:
        int: Cell Status
    """
    cell_status = None
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            # top left, top or top right
            if bbox[0, 1] < margin:
                # top left
                cell_status = 1
            elif bbox[1, 1] > patch_size - margin:
                # top right
                cell_status = 3
            else:
                # top
                cell_status = 2
        elif bbox[1, 1] > patch_size - margin:
            # top right, right or bottom right
            if bbox[1, 0] > patch_size - margin:
                # bottom right
                cell_status = 5
            else:
                # right
                cell_status = 4
        elif bbox[1, 0] > patch_size - margin:
            # bottom right, bottom, bottom left
            if bbox[0, 1] < margin:
                # bottom left
                cell_status = 7
            else:
                # bottom
                cell_status = 6
        elif bbox[0, 1] < margin:
            # bottom left, left, top left, but only left is left
            cell_status = 8
    else:
        cell_status = 0

    return cell_status

def get_edge_patch(position, row, col):
    # row starting on bottom or on top?
    if position == [1, 0, 0, 0]:
        # top
        return [[row - 1, col]]
    if position == [0, 1, 0, 0]:
        # right
        return [[row, col + 1]]
    if position == [0, 0, 1, 0]:
        # down
        return [[row + 1, col]]
    if position == [0, 0, 0, 1]:
        # left
        return [[row, col - 1]]
    if position == [1, 1, 0, 0]:
        # top and right
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1]]
    if position == [0, 0, 1, 1]:
        # down and left
        return [[row + 1, col], [row + 1, col - 1], [row, col - 1]]
    if position == [1, 0, 0, 1]:
        # left and top
        return [[row, col - 1], [row - 1, col - 1], [row - 1, col]]
    if position == [0, 1, 1, 0]:
        # right and down
        return [[row, col + 1], [row + 1, col + 1], [row + 1, col]]
    if position == [1, 0, 1, 0]:
        # top and down
        return [[row - 1, col], [row + 1, col]]
    if position == [0, 1, 0, 1]:
        # right and left
        return [[row, col + 1], [row, col - 1]]
    if position == [1, 1, 1, 0]:
        # top, right and down
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1], [row + 1, col + 1], [row + 1, col]]
    if position == [0, 1, 1, 1]:
        # right, down and left
        return [[row, col + 1], [row + 1, col + 1], [row + 1, col], [row + 1, col - 1], [row, col - 1]]
    if position == [1, 0, 1, 1]:
        # top, down and left
        return [[row - 1, col], [row - 1, col - 1], [row, col - 1], [row + 1, col - 1], [row + 1, col]]
    if position == [1, 1, 0, 1]:
        # top, right and left
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1], [row, col - 1], [row - 1, col - 1]]
    if position == [1, 1, 1, 1]:
        # top, right, down and left
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1], [row + 1, col + 1], [row + 1, col], [row + 1, col - 1], [row, col - 1], [row - 1, col - 1]]

def convert_geojson(cell_list: list[dict], polygons: bool = False) -> List[dict]:
    """Convert a list of cells to a geojson object

    Either a segmentation object (polygon) or detection points are converted

    Args:
        cell_list (list[dict]): Cell list with dict entry for each cell.
            Required keys for detection:
                * type
                * centroid
            Required keys for segmentation:
                * type
                * contour
        polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

    Returns:
        List[dict]: Geojson like list
    """
    if polygons:
        cell_segmentation_df = pd.DataFrame(cell_list)
        detected_types = sorted(cell_segmentation_df.type.unique())
        geojson_placeholder = []
        for cell_type in detected_types:
            cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
            contours = cells["contour"].to_list()
            final_c = []
            for c in contours:
                c.append(c[0])
                final_c.append([c])

            cell_geojson_object = get_template_segmentation()
            cell_geojson_object["id"] = str(uuid.uuid4())
            cell_geojson_object["geometry"]["coordinates"] = final_c
            cell_geojson_object["properties"]["classification"][
                "name"
            ] = TYPE_NUCLEI_DICT[cell_type]
            cell_geojson_object["properties"]["classification"][
                "color"
            ] = COLOR_DICT[cell_type]
            geojson_placeholder.append(cell_geojson_object)
    else:
        cell_detection_df = pd.DataFrame(cell_list)
        detected_types = sorted(cell_detection_df.type.unique())
        geojson_placeholder = []
        for cell_type in detected_types:
            cells = cell_detection_df[cell_detection_df["type"] == cell_type]
            centroids = cells["centroid"].to_list()
            cell_geojson_object = get_template_point()
            cell_geojson_object["id"] = str(uuid.uuid4())
            cell_geojson_object["geometry"]["coordinates"] = centroids
            cell_geojson_object["properties"]["classification"][
                "name"
            ] = TYPE_NUCLEI_DICT[cell_type]
            cell_geojson_object["properties"]["classification"][
                "color"
            ] = COLOR_DICT[cell_type]
            geojson_placeholder.append(cell_geojson_object)
    return geojson_placeholder


# CLI
class InferenceCellViTParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference for given run-directory with model checkpoints and logs",
        )

        parser.add_argument(
            "--config",
            type=str,
            help="Logging directory of a training run.",
            required=True,
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            help="Path to configuration file.",
            default="/path-to-CellVTA/CellVTA/logs/inference",
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference", default=1
        )
        parser.add_argument(
            "--plots",
            action="store_true",
            help="Generate inference plots in run_dir",
        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = InferenceCellViTParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)
    inf = InferenceCellViTUpscale(
        outdir=configuration["output_dir"],
        gpu=configuration["gpu"],
        config_path=configuration["config"],
        magnification=40,
    )
    inf.run_patch_inference(generate_plots=configuration["plots"])
    
