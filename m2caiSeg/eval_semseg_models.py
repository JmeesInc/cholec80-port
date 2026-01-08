"""
Evaluation script for semantic segmentation models (Unet, DeepLabV3, Segformer)
on GynSurg and m2caiSeg datasets.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall

class CFG:
    dataset_dir = Path("../cholec80-port-dataset/m2caiSeg_cleaned")
    test_image_dir = dataset_dir / "test_new" / "images"
    test_mask_dir = dataset_dir / "test_new" / "groundtruth"
    models_config = [
        {
            "name": "Unet (ConvNeXt-Base) - m2caiSeg",
            "type": "unet",
            "encoder": "tu-convnext_base",
            "checkpoint": "output/best.pth"
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
            "checkpoint": "../cholec80-port/output/best.pth"
        },
    ]

class TrocarsDataset(Dataset):
    """
    Trocarsのみを学習するためのDataset
    マスクファイルからtrocars（色=[170,85,85]）のみを抽出してバイナリマスクに変換
    """
    def __init__(self, image_dir, mask_dir, transform=None, trocars_color=None):
        """
        Args:
            image_dir: 画像ディレクトリのパス
            mask_dir: マスクディレクトリのパス
            transform: albumentations transform
            trocars_color: trocarsのRGB色 [R, G, B]
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.trocars_color = trocars_color if trocars_color is not None else CFG.trocars_color

        # 画像ファイルのリストを取得
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        
        # 対応するマスクファイルを確認
        self.valid_pairs = []
        for img_file in self.image_files:
            # マスクファイル名を取得（例: 00.jpg -> 00_gt.png）
            mask_name = img_file.stem + "_gt.png"
            mask_file = self.mask_dir / mask_name
            if mask_file.exists():
                self.valid_pairs.append((img_file, mask_file))
            else:
                self.valid_pairs.append((img_file, None))
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]

        # 画像を読み込み
        image = np.array(Image.open(img_path).convert("RGB"))

        # マスクを読み込み
        if mask_path is not None:
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        else:
            mask_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # trocarsのみを抽出してバイナリマスクに変換
        # 色が一致するピクセルを1、それ以外を0に
        trocars_mask = np.all(mask_rgb == self.trocars_color, axis=2).astype(np.float32)

        # データ拡張を適用
        if self.transform:
            augmented = self.transform(image=image, mask=trocars_mask)
            image = augmented['image']
            trocars_mask = augmented['mask']

        # マスクの次元を追加 (H, W) -> (1, H, W)
        if isinstance(trocars_mask, torch.Tensor):
            trocars_mask = trocars_mask.unsqueeze(0)
        elif isinstance(trocars_mask, np.ndarray):
            trocars_mask = torch.from_numpy(trocars_mask).unsqueeze(0)

        return image, trocars_mask

def get_val_transforms(size=384):
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def evaluate(model, dataloader, device, model_type, ds_name):
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
    # Dice for images that actually contain GT foreground
    dice_positive_only = []
    # Image-level detection counts (presence/absence)
    detection_counts = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}

    eps = 1e-8
    with torch.inference_mode():
        for i, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
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

            # Dice on images that have GT foreground
            gt_has_fg = masks.bool().any(dim=(1,2,3))
            if gt_has_fg.any():
                dice_positive_only.append(batch_stats['dice'][gt_has_fg].cpu())

            # Image-level detection (presence/absence) metrics
            pred_has_fg = probs.bool().any(dim=(1,2,3))
            detection_counts['tp'] += (pred_has_fg & gt_has_fg).sum().item()
            detection_counts['fp'] += (pred_has_fg & (~gt_has_fg)).sum().item()
            detection_counts['fn'] += ((~pred_has_fg) & gt_has_fg).sum().item()

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
    # Dice for GT-positive images only
    if dice_positive_only:
        vals = torch.cat(dice_positive_only, dim=0)
        stats['dice_pos_only'] = {
            "mean": vals.mean().item(),
            "std": vals.std(unbiased=False).item()
        }
    else:
        stats['dice_pos_only'] = {"mean": None, "std": None}

    # Image-level detection metrics (presence/absence)
    tp, fp, fn = detection_counts['tp'], detection_counts['fp'], detection_counts['fn']
    det_precision = tp / (tp + fp + eps)
    det_recall = tp / (tp + fn + eps)
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall + eps)
    stats['det_precision'] = {"mean": det_precision, "std": None}
    stats['det_recall'] = {"mean": det_recall, "std": None}
    stats['det_f1'] = {"mean": det_f1, "std": None}

    return stats

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []

    # Prepare datasets
    datasets = {
        "m2caiSeg (Test)": TrocarsDataset(
        CFG.test_image_dir,
        CFG.test_mask_dir,
        transform=get_val_transforms(),
        trocars_color=[170, 85, 85]
    )
    }

    dataloaders = {
        name: DataLoader(ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
        for name, ds in datasets.items()
    }

    for config in CFG.models_config:
        print(f"\nEvaluating Model: {config['name']}")
        checkpoint_path = Path(config['checkpoint'])
        
        if not checkpoint_path.exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            continue

        # Load model
        model = smp.Unet(encoder_name=config['encoder'], in_channels=3, classes=1, activation=None)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(device)

        # Evaluate on each dataset
        for ds_name, loader in dataloaders.items():
            print(f"  Dataset: {ds_name}")
            metrics = evaluate(model, loader, device, config['type'], ds_name)
            print(f"    IoU: {metrics['iou']['mean']:.4f} ± {metrics['iou']['std']:.4f}")
            print(f"    Dice: {metrics['dice']['mean']:.4f} ± {metrics['dice']['std']:.4f}")
            if metrics['dice_pos_only']['mean'] is not None:
                print(f"    Dice (GT>0): {metrics['dice_pos_only']['mean']:.4f} ± {metrics['dice_pos_only']['std']:.4f}")
            else:
                print(f"    Dice (GT>0): N/A (no positive GT images)")
            print(f"    Binary IoU: {metrics['binary_iou']['mean']:.4f} ± {metrics['binary_iou']['std']:.4f}")
            print(f"    Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
            print(f"    Recall: {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
            print(f"    Detection Precision: {metrics['det_precision']['mean']:.4f}")
            print(f"    Detection Recall: {metrics['det_recall']['mean']:.4f}")
            print(f"    Detection F1: {metrics['det_f1']['mean']:.4f}")
            
            results.append({
                "Model": config['name'],
                "Dataset": ds_name,
                "IoU Mean": metrics['iou']['mean'],
                "IoU Std": metrics['iou']['std'],
                "Dice Mean": metrics['dice']['mean'],
                "Dice Std": metrics['dice']['std'],
                "Dice (GT>0) Mean": metrics['dice_pos_only']['mean'],
                "Dice (GT>0) Std": metrics['dice_pos_only']['std'],
                "Binary IoU Mean": metrics['binary_iou']['mean'],
                "Binary IoU Std": metrics['binary_iou']['std'],
                "Precision Mean": metrics['precision']['mean'],
                "Precision Std": metrics['precision']['std'],
                "Recall Mean": metrics['recall']['mean'],
                "Recall Std": metrics['recall']['std'],
                "Det Precision": metrics['det_precision']['mean'],
                "Det Recall": metrics['det_recall']['mean'],
                "Det F1": metrics['det_f1']['mean'],
            })

    # Display summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    df = pd.DataFrame(results)
    if not df.empty:
        # Pivot table for better readability
        pivot_df = df.pivot(index='Model', columns='Dataset', values=[
            'IoU Mean', 'IoU Std',
            'Dice Mean', 'Dice Std',
            'Dice (GT>0) Mean', 'Dice (GT>0) Std',
            'Binary IoU Mean', 'Binary IoU Std',
            'Precision Mean', 'Precision Std',
            'Recall Mean', 'Recall Std',
            'Det Precision', 'Det Recall', 'Det F1'
        ])
        print(pivot_df)
        
        # Save to CSV
        df.to_csv("evaluation_results.csv", index=False)
        print("\nResults saved to evaluation_results.csv")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()

