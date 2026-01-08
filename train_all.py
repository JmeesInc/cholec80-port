"""
3つのデータセット（cholec80-port, GynSurg, m2caiSeg）を統合して
インカニュラ/トロカールのバイナリセグメンテーションを学習するスクリプト。
各元スクリプトの train split を train に、val split を val に利用する。
テストは実行しない。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU, DiceScore
import argparse


class CFG:
    # Model config
    encoder_name = "tu-convnext_base"
    in_channels = 3
    num_classes = 1  # Binary segmentation
    activation = None  # Sigmoid is applied in loss/metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training config
    batch_size = 16
    num_epochs = 50
    learning_rate = 5e-5
    num_workers = 32

    # Image config
    final_size = 384
    mask_threshold = 64

    # cholec80-port
    cholec_dataset_root = Path("cholec80-port-dataset/cholec80-port")
    cholec_train_videos = [f"video{i:02d}" for i in range(1, 9)] + [
        f"video{i:02d}_neg" for i in range(1, 9)
    ]
    cholec_val_videos = [f"video{i:02d}" for i in range(9, 11)] + [
        f"video{i:02d}_neg" for i in range(9, 11)
    ]

    # GynSurg
    gyn_dataset_root = Path("cholec80-port-dataset/GynSurg_cleaned")
    gyn_images_dir = gyn_dataset_root / "input"
    gyn_labels_dir = gyn_dataset_root / "new_mask"
    gyn_train_videos = [f"INSSEG_{i:02d}" for i in range(1, 9)]
    gyn_val_videos = [f"INSSEG_{i:02d}" for i in range(9, 11)]
    
    # m2caiSeg (trocars)
    m2cai_dataset_root = Path("cholec80-port-dataset/m2caiSeg_cleaned")
    m2cai_train_image_dir = m2cai_dataset_root / "train_new/images"
    m2cai_train_mask_dir = m2cai_dataset_root / "train_new/groundtruth"
    m2cai_val_image_dir = m2cai_dataset_root / "test_new/images"
    m2cai_val_mask_dir = m2cai_dataset_root / "test_new/groundtruth"

    # Output config
    checkpoint_dir = Path("_output_train_all")

def load_mask_from_file(mask_path, mask_threshold=64):
    """グレースケールマスクを2値化して読み込む。存在しない場合はゼロ配列を返す。"""
    if mask_path is None or not Path(mask_path).exists():
        return np.zeros((384, 384), dtype=np.uint8)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((384, 384), dtype=np.uint8)

    mask_binary = (mask > mask_threshold).astype(np.uint8)
    return mask_binary


class CholecPortDataset(Dataset):
    """cholec80-port-dataset 用 Dataset"""

    def __init__(self, dataset_root, video_names, transform=None, mask_threshold=64):
        self.dataset_root = Path(dataset_root)
        self.video_names = video_names
        self.transform = transform
        self.mask_threshold = mask_threshold

        self.valid_pairs = []
        for video_name in video_names:
            video_dir = self.dataset_root / video_name
            frame_dir = video_dir / "frame"
            mask_dir = video_dir / "mask"
            if not frame_dir.exists():
                continue
            for frame_file in sorted(frame_dir.glob("*.png")):
                mask_file = mask_dir / frame_file.name
                self.valid_pairs.append((frame_file, mask_file))

        print(f"[Cholec] Found {len(self.valid_pairs)} pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        mask = load_mask_from_file(mask_path, self.mask_threshold).astype(np.float32)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class GynSurgDataset(Dataset):
    """GynSurg INSSEG フォーマットの Dataset"""

    def __init__(
        self,
        images_root,
        masks_root=None,
        sequences=None,
        transform=None,
        mask_threshold=64,
        mask_suffix="_mask",
    ):
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root) if masks_root else None
        self.sequences = sequences
        self.transform = transform
        self.mask_threshold = mask_threshold
        self.mask_suffix = mask_suffix

        self.valid_pairs = []
        seq_list = (
            sequences
            if sequences is not None
            else [p.name for p in self.images_root.iterdir() if p.is_dir()]
        )
        for seq in sorted(seq_list):
            seq_img_dir = self.images_root / seq
            if not seq_img_dir.exists():
                continue
            for video_dir in sorted(seq_img_dir.glob("*")):
                if not video_dir.is_dir():
                    continue
                for frame_file in sorted(video_dir.glob("*.png")):
                    mask_file = None
                    if self.masks_root:
                        mask_file = (
                            self.masks_root
                            / seq
                            / video_dir.name
                            / f"{frame_file.stem}{self.mask_suffix}{frame_file.suffix}"
                        )
                    self.valid_pairs.append((frame_file, mask_file))

        print(f"[GynSurg] Found {len(self.valid_pairs)} pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        if mask_path is None or not mask_path.exists():
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = load_mask_from_file(mask_path, self.mask_threshold)

        mask = mask.astype(np.float32)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class M2caiTrocarsDataset(Dataset):
    """m2caiSeg trocars をバイナリ化する Dataset"""

    def __init__(self, image_dir, mask_dir, transform=None, trocars_color=np.array([170, 85, 85], dtype=np.uint8)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.trocars_color = trocars_color

        image_files = sorted(
            list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        )
        self.valid_pairs = []
        for img_file in image_files:
            mask_file = self.mask_dir / f"{img_file.stem}_gt.png"
            if mask_file.exists():
                self.valid_pairs.append((img_file, mask_file))
            else:
                self.valid_pairs.append((img_file, None))

        print(f"[m2caiSeg] Found {len(self.valid_pairs)} pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        if mask_path is not None and mask_path.exists():
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        else:
            mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        mask = np.all(mask_rgb == self.trocars_color, axis=2).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def get_train_transforms(cfg):
    """Training data augmentation (from 3_train_v1.0_semseg.py)"""
    return A.Compose(
        [
            A.Resize(cfg.final_size, cfg.final_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.5, 2.0), rotate_limit=180, p=0.8),
            A.RandomBrightnessContrast(p=0.5),
            A.FancyPCA(p=0.5),
            A.OneOf(
                [
                    A.RandomFog(fog_coef_range=(0.3, 0.5)),
                    A.RandomSunFlare(),
                    A.RandomShadow(),
                ],
                p=0.2,
            ),
            A.MotionBlur(blur_limit=(3, 15), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


def get_val_transforms(cfg):
    """Validation transforms"""
    return A.Compose(
        [
            A.Resize(height=cfg.final_size, width=cfg.final_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(pred, target, threshold=0.5):
    """IoU, Dice, Precision, Recall を計算"""
    with torch.no_grad():
        probs = torch.sigmoid(pred)
        pred_idx = (probs > threshold).long()
        target_idx = target.long()

        metrics = MetricCollection(
            {
                "iou": MeanIoU(num_classes=2, input_format="index"),
                "dice": DiceScore(
                    num_classes=2, input_format="index", average="macro"
                ),
            }
        ).to(pred.device)
        metric_vals = metrics(pred_idx, target_idx)

        tp = ((pred_idx == 1) & (target_idx == 1)).sum().item()
        fp = ((pred_idx == 1) & (target_idx == 0)).sum().item()
        fn = ((pred_idx == 0) & (target_idx == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

    return {
        "iou": metric_vals["iou"].item(),
        "dice": metric_vals["dice"].item(),
        "precision": precision,
        "recall": recall,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        metrics = calculate_metrics(outputs, masks)
        running_loss += loss.item()
        for k in running_metrics:
            running_metrics[k] += metrics[k]

        pbar.set_postfix({"loss": loss.item(), "iou": metrics["iou"], "dice": metrics["dice"]})

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    running_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            metrics = calculate_metrics(outputs, masks)

            running_loss += loss.item()
            for k in running_metrics:
                running_metrics[k] += metrics[k]

            pbar.set_postfix({"loss": loss.item(), "iou": metrics["iou"], "dice": metrics["dice"]})

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    return avg_loss, avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Train on all datasets (combined)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    cfg = CFG()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("Combined Training (cholec80-port + GynSurg + m2caiSeg)")
    print(f"Device: {cfg.device}")
    print(f"Model: smp.Unet({cfg.encoder_name}, num_classes={cfg.num_classes})")
    print(f"Image size: {cfg.final_size}x{cfg.final_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("=" * 60)

    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    # Datasets
    cholec_train = CholecPortDataset(
        cfg.cholec_dataset_root,
        cfg.cholec_train_videos,
        transform=train_transform,
        mask_threshold=cfg.mask_threshold,
    )
    cholec_val = CholecPortDataset(
        cfg.cholec_dataset_root,
        cfg.cholec_val_videos,
        transform=val_transform,
        mask_threshold=cfg.mask_threshold,
    )

    gyn_train = GynSurgDataset(
        cfg.gyn_images_dir,
        cfg.gyn_labels_dir,
        sequences=cfg.gyn_train_videos,
        transform=train_transform,
        mask_threshold=cfg.mask_threshold,
    )
    gyn_val = GynSurgDataset(
        cfg.gyn_images_dir,
        cfg.gyn_labels_dir,
        sequences=cfg.gyn_val_videos,
        transform=val_transform,
        mask_threshold=cfg.mask_threshold,
    )

    m2cai_train = M2caiTrocarsDataset(
        cfg.m2cai_train_image_dir,
        cfg.m2cai_train_mask_dir,
        transform=train_transform,
    )
    m2cai_val = M2caiTrocarsDataset(
        cfg.m2cai_val_image_dir,
        cfg.m2cai_val_mask_dir,
        transform=val_transform,
    )

    train_dataset = ConcatDataset([cholec_train, gyn_train, m2cai_train])
    val_dataset = ConcatDataset([cholec_val, gyn_val, m2cai_val])

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)} (cholec {len(cholec_train)}, gyn {len(gyn_train)}, m2cai {len(m2cai_train)})")
    print(f"Val samples:   {len(val_dataset)} (cholec {len(cholec_val)}, gyn {len(gyn_val)}, m2cai {len(m2cai_val)})")
    print("=" * 60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        in_channels=cfg.in_channels,
        classes=cfg.num_classes,
        activation=cfg.activation,
    ).to(cfg.device)

    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
    }

    start_epoch = 0
    best_val_iou = 0.0
    if args.resume:
        checkpoint_path = cfg.checkpoint_dir / "latest.pth"
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            history = checkpoint.get("history", history)
            best_val_iou = (
                max(history.get("val_iou", [0.0])) if history.get("val_iou") else 0.0
            )
            print(f"Resume from epoch {start_epoch}, best Val IoU {best_val_iou:.4f}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")

    print("\nStarting training...")
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        print("-" * 60)

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, epoch
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, cfg.device, epoch
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

        print(
            f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            f"Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}\n"
            f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}\n"
            f"Train Prec: {train_metrics['precision']:.4f} | Val Prec: {val_metrics['precision']:.4f}\n"
            f"Train Rec: {train_metrics['recall']:.4f} | Val Rec: {val_metrics['recall']:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "history": history,
        }
        torch.save(checkpoint, cfg.checkpoint_dir / "latest.pth")
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(checkpoint, cfg.checkpoint_dir / "best.pth")
            print(f"✅ Best model saved (Val IoU {best_val_iou:.4f})")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

