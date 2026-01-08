"""
Trocarsのみを学習するスクリプト
モデル: smp.Unet("tu-convnext_base", num_classes=1)
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
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
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU, DiceScore

class CFG:
    # Model config
    encoder_name = "tu-convnext_base"
    in_channels = 3
    num_classes = 1  # Binary segmentation for trocars
    activation = None  # Will use sigmoid for binary
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training config
    batch_size = 8
    num_epochs = 50
    learning_rate = 5e-5
    num_workers = 16

    # Image config
    final_size = 384

    # Data paths
    dataset_dir = Path("../cholec80-port-dataset/m2caiSeg_cleaned")
    train_image_dir = dataset_dir / "train_new" / "images"
    train_mask_dir = dataset_dir / "train_new" / "groundtruth"
    val_image_dir = dataset_dir / "test_new" / "images"
    val_mask_dir = dataset_dir / "test_new" / "groundtruth"

    # Output config
    checkpoint_dir = Path("output")

    fill = False

    # Trocars color (RGB)
    trocars_color = np.array([170, 85, 85], dtype=np.uint8)



class TrocarsDataset(Dataset):
    """
    Trocarsのみを学習するためのDataset
    マスクファイルからtrocars（色=[170,85,85]）のみを抽出してバイナリマスクに変換
    """
    def __init__(self, image_dir, mask_dir, transform=None, trocars_color=None, fill=False):
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
        self.fill = fill

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
                # load zeros
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
            mask_rgb = np.zeros((image.shape[0], image.shape[1], 3))

        # trocarsのみを抽出してバイナリマスクに変換
        # 色が一致するピクセルを1、それ以外を0に
        trocars_mask = np.all(mask_rgb == self.trocars_color, axis=2).astype(np.float32)

        if self.fill:
            trocars_mask = (trocars_mask > 0.5).astype(np.uint8)
            # 塗りつぶし多角形を描写するためのゼロ埋め配列定義
            # point:opencvの関数で扱えるように型をuint8で指定！
            zero_img = np.zeros([trocars_mask.shape[0], trocars_mask.shape[1]], dtype="uint8")
            # 全ての輪郭座標配列を使って塗りつぶし多角形を描写
            contours,_ = cv2.findContours(trocars_mask, 1, 2)
            for p in contours:
                cv2.fillPoly(zero_img, [p], 1)
            trocars_mask = zero_img.astype(np.float32)
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

def get_train_transforms(cfg):
    """学習用のデータ拡張"""
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
    """検証用の変換"""
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
    """1エポックの学習"""
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
    """検証"""
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
    parser = argparse.ArgumentParser(description='Trocars Segmentation Training')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()

    cfg = CFG()

    # Create output directories
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("="*60)
    print("Trocars Segmentation Training")
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
    train_dataset = TrocarsDataset(
        cfg.train_image_dir,
        cfg.train_mask_dir,
        transform=get_train_transforms(cfg),
        trocars_color=cfg.trocars_color,
        fill=cfg.fill
    )
    val_dataset = TrocarsDataset(
        cfg.val_image_dir,
        cfg.val_mask_dir,
        transform=get_val_transforms(cfg),
        trocars_color=cfg.trocars_color,
        fill = cfg.fill
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
    }

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_iou = 0.0

    if args.resume:
        checkpoint_path = os.path.join(cfg.checkpoint_dir, 'latest.pth')
        if os.path.exists(checkpoint_path):
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
        torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, 'latest.pth'))

        # Save best checkpoint
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, 'best.pth'))
            print(f"✅ Best model saved with Val IoU: {best_val_iou:.4f}")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
