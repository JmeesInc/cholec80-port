"""
Evaluation script for semantic segmentation models (Unet, DeepLabV3, Segformer)
on GynSurg and m2caiSeg datasets.
"""

import os
# GPU settings (Adjust as needed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall

import json

class CFG:
    dataset_root = Path("../cholec80-port-dataset/cholec80-port")
    test_videos = [f"video{i:02d}" for i in range(11, 21)] + [f"video{i:02d}_neg" for i in range(11, 21)]

    models_config = [
        {
            "name": "Unet (ConvNeXt-Base) - m2caiSeg",
            "type": "unet",
            "encoder": "tu-convnext_base",
            "checkpoint": "../m2caiSeg/output/best.pth"
        },
        {
            "name": "Unet (ConvNeXt-Base) - GynSurg",
            "type": "unet",
            "encoder": "tu-convnext_base",
            "checkpoint": "../GynSurg/output/best.pth"
        },
        {
            "name": "Unet (ConvNeXt-Base) - cholec80-port",
            "type": "unet",
            "encoder": "tu-convnext_base",
            "checkpoint": "output/best.pth"
        },
        {
            "name": "Unet (ConvNeXt-Base) - cholec80-all",
            "type": "unet",
            "encoder": "tu-convnext_base",
            "checkpoint": "../output_train_all/best.pth"
        },
    ]

class Cholec80PortDataset(Dataset):
    """
    Dataset for Cholec80-Port and m2caiSeg evaluation.
    Based on m2caiSeg/train.py implementation.
    """
    def __init__(self, dataset_root, video_names, transform=None, mask_threshold=64):
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.video_names = video_names
        self.transform = transform
        self.mask_threshold = mask_threshold
        self.valid_pairs = []

        # For Cholec80-Port dataset
        for video_name in video_names:
            video_dir = self.dataset_root / video_name
            frame_dir = video_dir / "frame"
            mask_dir = video_dir / "mask"
            
            if not frame_dir.exists():
                print(f"Warning: Frame directory not found: {frame_dir}")
                continue
            
            for frame_file in sorted(frame_dir.glob("*.png")):
                # Mask file path (may not exist for negative samples)
                mask_file = mask_dir / frame_file.name
                self.valid_pairs.append((frame_file, mask_file))

        print(f"Found {len(self.valid_pairs)} valid image-label pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.valid_pairs[idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        img_height, img_width = image.shape[:2]

        # Load grayscale mask (Cholec80-Port style)
        if not label_path.exists():
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
        else:
            mask_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
            else:
                mask = (mask_img > self.mask_threshold).astype(np.uint8)
        
        mask = mask.astype(np.float32)
        
        # Resize mask if needed
        if mask.shape != (img_height, img_width):
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add channel dimension
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)

        meta = {
            "img_path": str(img_path),
            "label_path": str(label_path)
        }

        return image, mask, meta

def get_val_transforms(size=384):
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def evaluate(model, dataloader, device, model_type, ds_name, visualize_top_k: int = 5):
    model.eval()

    def compute_per_sample_metrics(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8):
        """
        preds: (B, 1, H, W) long/bool
        targets: (B, 1, H, W) long/bool
        returns dict of tensors shaped (B,)
        """
        preds_flat = preds.view(preds.shape[0], -1).bool()
        targets_flat = targets.view(targets.shape[0], -1).bool()

        tp = (preds_flat & targets_flat).sum(dim=1).float()
        fp = (preds_flat & ~targets_flat).sum(dim=1).float()
        fn = (~preds_flat & targets_flat).sum(dim=1).float()
        tn = (~preds_flat & ~targets_flat).sum(dim=1).float()

        # Foreground (tool) metrics
        binary_iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        # Mean IoU across foreground/background (2-class)
        bg_intersection = tn
        bg_union = tn + fp + fn
        bg_iou = bg_intersection / (bg_union + eps)
        mean_iou = (binary_iou + bg_iou) / 2

        return {
            "iou": mean_iou,
            "dice": dice,
            "binary_iou": binary_iou,
            "precision": precision,
            "recall": recall,
        }

    # TorchMetrics for segmentation (binary -> 2-class)
    metrics = MetricCollection({
        'iou': MeanIoU(num_classes=2, include_background=True, input_format='index'),
        'dice': DiceScore(num_classes=2, include_background=False, input_format='index', aggregation_level='global'),
        'binary_iou': BinaryJaccardIndex(),
        'precision': BinaryPrecision(),
        'recall': BinaryRecall(),
    }).to(device)

    # Per-sample values to compute std
    per_sample_values = {k: [] for k in ['iou', 'dice', 'binary_iou', 'precision', 'recall']}
    per_sample_records = []

    eps = 1e-8
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # batch may include metadata
            if len(batch) == 2:
                images, masks = batch
                metas = None
            else:
                images, masks, metas = batch

            images = images.to(device)
            masks = masks.to(device).float()
            
            outputs = model(images)

            # Standardize to probabilities
            probs = torch.sigmoid(outputs)
            probs = (probs>0.5).long()
            masks = masks.long()
            metrics.update(probs, masks)

            # Collect per-sample metrics for std computation
            batch_stats = compute_per_sample_metrics(probs, masks, eps)
            for key in per_sample_values:
                per_sample_values[key].append(batch_stats[key].cpu())

            # Track per-sample records for later visualization
            if metas is not None:
                # metas is a dict of lists due to default_collate
                meta_img_paths = metas.get("img_path", [])
                meta_label_paths = metas.get("label_path", [])
            else:
                meta_img_paths = [None] * masks.shape[0]
                meta_label_paths = [None] * masks.shape[0]

            for idx_in_batch in range(masks.shape[0]):
                per_sample_records.append({
                    "img_path": meta_img_paths[idx_in_batch],
                    "label_path": meta_label_paths[idx_in_batch],
                    "iou": batch_stats["iou"][idx_in_batch].item(),
                    "dice": batch_stats["dice"][idx_in_batch].item(),
                    "binary_iou": batch_stats["binary_iou"][idx_in_batch].item(),
                    "precision": batch_stats["precision"][idx_in_batch].item(),
                    "recall": batch_stats["recall"][idx_in_batch].item(),
                })

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            img = images[0].cpu().numpy()  # Take the first image in the batch
            mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
            img = img * std + mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            # Change from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1,2,0))
            ax[0].imshow(img)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            mask = masks[0, 0].cpu().numpy()
            mask = mask * 255
            mask = mask.astype(np.uint8)
            ax[1].imshow(mask)
            ax[1].set_title('Ground Truth')
            ax[1].axis('off')
            pred = probs[0, 0].cpu().numpy()
            pred = pred * 255
            pred = pred.astype(np.uint8)
            ax[2].imshow(pred)
            ax[2].set_title('Prediction')
            ax[2].axis('off')
            plt.savefig(f"output/{model_type}/{ds_name}_{i:03d}.png")
            plt.close()

    computed = metrics.compute()

    # Assemble mean/std dict
    stats = {}
    for key in per_sample_values:
        if per_sample_values[key]:
            vals = torch.cat(per_sample_values[key], dim=0)
            stats[key] = {
                "mean": computed[key].item(),
                "std": vals.std(unbiased=False).item()
            }
    return stats, per_sample_records


def _load_mask_for_visualization(label_path: str, use_rgb_mask: bool, target_color: np.ndarray, mask_threshold: int, desired_size=None):
    if label_path is None or not Path(label_path).exists():
        return None

    if use_rgb_mask:
        mask_rgb = np.array(Image.open(label_path).convert("RGB"))
        mask = np.all(mask_rgb == target_color, axis=2).astype(np.float32)
    else:
        mask_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            mask = None
        else:
            mask = (mask_img > mask_threshold).astype(np.float32)

    if mask is None:
        return None

    if desired_size is not None and mask.shape != desired_size:
        mask = cv2.resize(mask, (desired_size[1], desired_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _denormalize_image(img_tensor: torch.Tensor):
    # img_tensor: (3, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return img_np


def visualize_problematic_samples(model, device, dataset, per_sample_records, out_dir: Path, top_k: int = 5):
    """
    可視化: recall低, precision低, IoU高(良例)を保存する
    """
    if not per_sample_records:
        print("No per-sample records to visualize.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "low_recall": sorted(per_sample_records, key=lambda r: r["recall"])[:top_k],
        "low_precision": sorted(per_sample_records, key=lambda r: r["precision"])[:top_k],
        "high_iou": sorted(per_sample_records, key=lambda r: r["iou"], reverse=True)[:top_k],
    }

    use_rgb_mask = getattr(dataset, "use_rgb_mask", False)
    target_color = getattr(dataset, "target_color", np.array([170, 85, 85], dtype=np.uint8))
    mask_threshold = getattr(dataset, "mask_threshold", 64)
    transform = getattr(dataset, "transform", None)

    model_was_training = model.training
    model.eval()

    with torch.inference_mode():
        for category, samples in categories.items():
            for idx, rec in enumerate(samples):
                img_path = rec.get("img_path")
                label_path = rec.get("label_path")
                if img_path is None:
                    continue

                try:
                    image = np.array(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")
                    continue

                h, w = image.shape[:2]
                mask = _load_mask_for_visualization(label_path, use_rgb_mask, target_color, mask_threshold, desired_size=(h, w))

                if transform is not None:
                    transformed = transform(image=image, mask=mask if mask is not None else np.zeros((h, w), dtype=np.float32))
                    img_t = transformed["image"].unsqueeze(0).to(device)
                    mask_vis = transformed["mask"].cpu().numpy() if isinstance(transformed["mask"], torch.Tensor) else transformed["mask"]
                else:
                    # 最低限の変換
                    img_t = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
                    mask_vis = mask

                preds = model(img_t)
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).long()
                pred_np = preds[0, 0].cpu().numpy()

                img_vis = _denormalize_image(img_t[0])
                # 誤差可視化: 正解=緑, FP=赤, FN=青
                if mask_vis is None:
                    mask_bool = np.zeros_like(pred_np, dtype=bool)
                else:
                    mask_bool = mask_vis.astype(bool)
                pred_bool = pred_np.astype(bool)

                correct = pred_bool & mask_bool        # TP
                false_positive = pred_bool & ~mask_bool  # FP
                false_negative = ~pred_bool & mask_bool  # FN

                overlay = np.zeros_like(img_vis)
                # Green for correct
                overlay[..., 1] = np.where(correct, 255, 0).astype(np.uint8)
                # Red for FP
                overlay[..., 0] = np.where(false_positive, 255, 0).astype(np.uint8)
                # Blue for FN
                overlay[..., 2] = np.where(false_negative, 255, 0).astype(np.uint8)

                alpha = 0.6
                blended = cv2.addWeighted(img_vis, 1.0, overlay, alpha, 0)

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(img_vis)
                ax[0].set_title("Input")
                ax[0].axis("off")

                ax[1].imshow(mask_vis if mask_vis is not None else np.zeros_like(pred_np))
                ax[1].set_title("Ground Truth")
                ax[1].axis("off")

                ax[2].imshow(blended)
                ax[2].set_title("Pred (R) / GT (G)")
                ax[2].axis("off")

                fig.suptitle(
                    f"{category}: IoU {rec['iou']:.3f}, Dice {rec['dice']:.3f}, "
                    f"Prec {rec['precision']:.3f}, Recall {rec['recall']:.3f}\n"
                    f"{img_path}"
                )

                save_path = out_dir / f"{category}_{idx:03d}.png"
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close(fig)

    if model_was_training:
        model.train()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    results = []

    # Prepare datasets
    datasets = {
        "Cholec80-Port (Test: vid01-10)": Cholec80PortDataset(
            dataset_root=CFG.dataset_root,
            video_names=CFG.test_videos,
            transform=get_val_transforms(),
            mask_threshold=64,
        ),
    }

    dataloaders = {
        name: DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
        for name, ds in datasets.items()
    }

    for config in CFG.models_config:
        print(f"\nEvaluating Model: {config['name']}")
        checkpoint_path = Path(config['checkpoint'])
        
        if not checkpoint_path.exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            continue

        # Load model
        if config['type'] == 'unet':
            model = smp.Unet(encoder_name=config['encoder'], in_channels=3, classes=1, activation=None)
        elif config['type'] == 'deeplabv3':
            model = smp.DeepLabV3(encoder_name=config['encoder'], in_channels=3, classes=1, activation=None)
        elif config['type'] == 'segformer':
            model = smp.Segformer(encoder_name=config['encoder'], in_channels=3, classes=1, activation=None)
        
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(device)

        # Evaluate on each dataset
        for ds_name, loader in dataloaders.items():
            print(f"  Dataset: {ds_name}")
            metrics, per_sample_records = evaluate(model, loader, device, config['type'], ds_name, visualize_top_k=5)
            print(f"    IoU: {metrics['iou']['mean']:.4f} ± {metrics['iou']['std']:.4f}")
            print(f"    Dice: {metrics['dice']['mean']:.4f} ± {metrics['dice']['std']:.4f}")
            print(f"    Binary IoU: {metrics['binary_iou']['mean']:.4f} ± {metrics['binary_iou']['std']:.4f}")
            print(f"    Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
            print(f"    Recall: {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")

            # 可視化（recall低, precision低, IoU高）
            vis_dir = Path(f"pred_vis/{config['name']}/")
            visualize_problematic_samples(
                model=model,
                device=device,
                dataset=loader.dataset,
                per_sample_records=per_sample_records,
                out_dir=vis_dir,
                top_k=5
            )
            
            results.append({
                "Model": config['name'],
                "Dataset": ds_name,
                "IoU Mean": metrics['iou']['mean'],
                "IoU Std": metrics['iou']['std'],
                "Dice Mean": metrics['dice']['mean'],
                "Dice Std": metrics['dice']['std'],
                "Binary IoU Mean": metrics['binary_iou']['mean'],
                "Binary IoU Std": metrics['binary_iou']['std'],
                "Precision Mean": metrics['precision']['mean'],
                "Precision Std": metrics['precision']['std'],
                "Recall Mean": metrics['recall']['mean'],
                "Recall Std": metrics['recall']['std'],
            })

    # Display summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)

if __name__ == "__main__":
    main()

