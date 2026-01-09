"""
Semantic segmentation training for in-cannula v1.0.

Dataset:
- train: video01-08
- val: video09-10 (annotated)

Model: smp.Unet("tu-convnext_base", num_classes=1)
Based on m2caiSeg/train.py

Usage:
    python train_v1.0_semseg.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
import argparse
import cv2
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU, DiceScore


class CFG:
    # Model config
    encoder_name = "tu-convnext_base"
    in_channels = 3
    num_classes = 1  # Binary segmentation for in-cannula
    activation = None  # Will use sigmoid for binary
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training config
    batch_size = 16
    num_epochs = 50
    learning_rate = 5e-5
    num_workers = 16

    # Image config
    final_size = 384
    mask_threshold = 64
    mask_suffix = "_mask"

    # Data paths
    dataset_root = Path("../cholec80-port-dataset/GynSurg_cleaned")

    train_images_dir = dataset_root / "input"
    train_labels_dir = dataset_root / "new_mask"
    val_images_dir = dataset_root / "input"
    val_labels_dir = dataset_root / "new_mask"
    test_images_dir = dataset_root / "input"
    test_labels_dir = dataset_root / "new_mask"
    train_videos = [f"INSSEG_{i:02d}" for i in range(1, 9)]
    val_videos = [f"INSSEG_{i:02d}" for i in range(9, 11)]
    test_videos = [f"INSSEG_{i:02d}" for i in range(9, 11)]
    # Output config
    checkpoint_dir = Path("output")

def load_mask_from_file(mask_path, mask_threshold=64):
    """
    Load binary mask from grayscale mask file.

    Args:
        mask_path: Path to mask image file (.png)
        mask_threshold: Threshold for binarization

    Returns:
        numpy.ndarray: Binary mask (H, W) with values 0 or 1
    """
    if not mask_path.exists():
        return np.zeros((384, 384), dtype=np.uint8)  # Default size
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((384, 384), dtype=np.uint8)
    
    # Binarize mask (same as prepare_yolo_dataset.py)
    mask_binary = (mask > mask_threshold).astype(np.uint8)
    return mask_binary


def rgb_mask_to_binary(mask_path):
    """
    Convert RGB mask image to binary mask (for m2caiSeg test dataset).
    Extracts in-cannula pixels (color=[170, 85, 85] based on m2caiSeg format).

    Args:
        mask_path: Path to RGB mask image

    Returns:
        numpy.ndarray: Binary mask (H, W) with values 0 or 1
    """
    if not mask_path.exists():
        return np.zeros((384, 384), dtype=np.uint8)  # Default size

    mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
    # Extract in-cannula pixels (assuming same color scheme as m2caiSeg)
    # For now, convert any non-black pixels to 1 (binary segmentation)
    mask_binary = (np.sum(mask_rgb, axis=2) > 0).astype(np.uint8)
    return mask_binary



class InCannulaDataset(Dataset):
    """
    Dataset for in-cannula semantic segmentation.
    Loads images and masks from INSSEG-style folders under images_root/masks_root.
    """
    def __init__(
        self,
        images_root,
        masks_root=None,
        sequences=None,
        transform=None,
        mask_threshold=64,
        mask_suffix="_mask",
        use_rgb_mask=False,
        rgb_mask_dir=None,
    ):
        """
        Args:
            images_root: Root dir that contains INSSEG_* sequence folders.
            masks_root: Root dir mirroring images_root that stores *_mask.png.
            sequences: List of sequence folder names to include (e.g., ["INSSEG_01"]).
                       If None, all sequences under images_root are used.
            transform: albumentations transform
            mask_threshold: Threshold for mask binarization
            mask_suffix: Suffix appended before extension for mask filenames
            use_rgb_mask: If True, load RGB masks from rgb_mask_dir (external test set)
            rgb_mask_dir: Directory for RGB mask images (m2caiSeg-style test)
        """
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root) if masks_root else None
        self.sequences = sequences
        self.transform = transform
        self.mask_threshold = mask_threshold
        self.mask_suffix = mask_suffix
        self.use_rgb_mask = use_rgb_mask
        self.rgb_mask_dir = Path(rgb_mask_dir) if rgb_mask_dir else None

        # Collect all image-mask pairs
        self.valid_pairs = []

        if self.use_rgb_mask and self.rgb_mask_dir:
            image_dir = self.rgb_mask_dir.parent / "images"
            for img_file in sorted(image_dir.glob("*.jpg")):
                mask_file = self.rgb_mask_dir / f"{img_file.stem}_gt.png"
                if mask_file.exists():
                    self.valid_pairs.append((img_file, mask_file))
        else:
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

        print(f"Found {len(self.valid_pairs)} valid image-label pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        img_height, img_width = image.shape[:2]

        # Load mask
        if self.use_rgb_mask and self.rgb_mask_dir:
            mask = rgb_mask_to_binary(mask_path).astype(np.float32)
        else:
            if mask_path is None:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
            else:
                mask = load_mask_from_file(mask_path, self.mask_threshold)
                

        # Align mask size to image size to satisfy Albumentations shape check
        if mask.shape != (img_height, img_width):
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add channel dimension to mask (H, W) -> (1, H, W)
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def get_train_transforms(cfg):
    """Training data augmentation"""
    return A.Compose(
        [
            A.Resize(cfg.final_size, cfg.final_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                border_mode=0, rotate_limit=90, scale_limit=0.5, shift_limit=0.2
            ),
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
            A.CoarseDropout(max_holes=10, max_height=100, max_width=100, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


def get_val_transforms(cfg):
    """Validation transforms"""
    return A.Compose([
        A.Resize(height=cfg.final_size, width=cfg.final_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """BCE Loss + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate evaluation metrics

    Args:
        pred: predicted logits (B, 1, H, W)
        target: ground truth masks (B, 1, H, W)
        threshold: threshold for binary prediction

    Returns:
        dict: metrics including IoU, Dice, Precision, Recall
    """
    with torch.no_grad():
        probs = torch.sigmoid(pred)
        pred_idx = (probs > threshold).long()
        target_idx = target.long()

        metrics = MetricCollection({
            'iou': MeanIoU(num_classes=2, input_format='index'),
            'dice': DiceScore(num_classes=2, input_format='index', average='macro'),
        }).to(pred.device)
        metric_vals = metrics(pred_idx, target_idx)

        tp = ((pred_idx == 1) & (target_idx == 1)).sum().item()
        fp = ((pred_idx == 1) & (target_idx == 0)).sum().item()
        fn = ((pred_idx == 0) & (target_idx == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

    return {
        'iou': metric_vals['iou'].item(),
        'dice': metric_vals['dice'].item(),
        'precision': precision,
        'recall': recall
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    running_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        metrics = calculate_metrics(outputs, masks)

        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += metrics[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'iou': metrics['iou'],
            'dice': metrics['dice'],
        })

    # Calculate average
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_metrics = {key: val / num_batches for key, val in running_metrics.items()}

    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device, epoch):
    """Validation"""
    model.eval()

    running_loss = 0.0
    running_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Metrics
            metrics = calculate_metrics(outputs, masks)

            running_loss += loss.item()
            for key in running_metrics:
                running_metrics[key] += metrics[key]

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'iou': metrics['iou'],
                'dice': metrics['dice'],
            })

    # Calculate average
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_metrics = {key: val / num_batches for key, val in running_metrics.items()}

    return avg_loss, avg_metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='In-Cannula Semantic Segmentation Training v1.0')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()

    cfg = CFG()

    # Create output directories
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("="*60)
    print("In-Cannula Semantic Segmentation Training v1.0")
    if args.resume:
        print("MODE: Resume from latest checkpoint")
    else:
        print("MODE: Train from scratch")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"Model: smp.Unet({cfg.encoder_name}, num_classes={cfg.num_classes})")
    print(f"Image size: {cfg.final_size}x{cfg.final_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("="*60)

    # Create datasets
    train_dataset = InCannulaDataset(
        cfg.train_images_dir,
        cfg.train_labels_dir,
        sequences=cfg.train_videos,
        transform=get_train_transforms(cfg),
        mask_threshold=cfg.mask_threshold,
        mask_suffix=cfg.mask_suffix,
    )
    val_dataset = InCannulaDataset(
        cfg.val_images_dir,
        cfg.val_labels_dir,
        sequences=cfg.val_videos,
        transform=get_val_transforms(cfg),
        mask_threshold=cfg.mask_threshold,
        mask_suffix=cfg.mask_suffix,
    )

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"{'='*60}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    # Create model
    print(f"\nCreating model: smp.Unet({cfg.encoder_name}, num_classes={cfg.num_classes})")
    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        in_channels=cfg.in_channels,
        classes=cfg.num_classes,
        activation=cfg.activation,
    ).to(cfg.device)

    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'test_loss': [],
        'test_iou': [],
        'test_dice': [],
        'test_precision': [],
        'test_recall': [],
        'test2_loss': [],
        'test2_iou': [],
        'test2_dice': [],
        'test2_precision': [],
        'test2_recall': [],
    }

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_iou = 0.0

    if args.resume:
        checkpoint_path = cfg.checkpoint_dir / 'latest.pth'
        if checkpoint_path.exists():
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            history = checkpoint.get('history', history)
            best_val_iou = max(history.get('val_iou', [0.0])) if history.get('val_iou') else 0.0
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Resuming from epoch: {start_epoch}")
            print(f"   Best Val IoU so far: {best_val_iou:.4f}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")

    # Training loop
    print("\nStarting training...")
    print(f"Training from epoch {start_epoch + 1} to {cfg.num_epochs}")

    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        print("-" * 60)

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, epoch
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, cfg.device, epoch
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        print(f"Train Prec: {train_metrics['precision']:.4f} | Val Prec: {val_metrics['precision']:.4f}")
        print(f"Train Rec: {train_metrics['recall']:.4f} | Val Rec: {val_metrics['recall']:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'history': history,
        }

        # Save latest checkpoint
        torch.save(checkpoint, cfg.checkpoint_dir / 'latest.pth')

        # Save best checkpoint
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(checkpoint, cfg.checkpoint_dir / 'best.pth')
            print(f"✅ Best model saved with Val IoU: {best_val_iou:.4f}")
    
    test_loss, test_metrics = validate(
        model, val_loader, criterion, cfg.device, epoch
    )
    history['test_loss'].append(test_loss)
    history['test_iou'].append(test_metrics['iou'])
    history['test_dice'].append(test_metrics['dice'])
    history['test_precision'].append(test_metrics['precision'])
    history['test_recall'].append(test_metrics['recall'])
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_metrics['iou']:.4f} | Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test Prec: {test_metrics['precision']:.4f} | Test Rec: {test_metrics['recall']:.4f}")
    print("="*60)

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

