from pathlib import Path
import cv2
import numpy as np

# 設定値
DATASET_ROOT = Path("../cholec80-port-dataset/cholec80-port")
CLASS_NAME = "port"
CLASS_COLOR_BGR = (255, 255, 0)  # Cyan
MASK_THRESHOLD = 64  # prepare_yolo_dataset.py と同じ
MAX_IMAGES = 300  # 保存する最大枚数

# 出力先
base_output_dir = Path("../vis/cholec80-port")
output_dir = base_output_dir / CLASS_NAME
output_dir.mkdir(parents=True, exist_ok=True)


def collect_image_mask_pairs(dataset_root: Path):
    """frame/mask を動画ごとに収集する."""
    pairs = []
    for video_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        frame_dir = video_dir / "frame"
        mask_dir = video_dir / "mask"
        if not frame_dir.exists():
            continue
        for frame_file in sorted(frame_dir.glob("*.png")):
            mask_file = mask_dir / frame_file.name
            pairs.append((frame_file, mask_file, video_dir.name))
    return pairs


def load_binary_mask(mask_path: Path, target_shape):
    """グレースケールマスクを 0/1 の2値化で読み込む."""
    mask = np.zeros(target_shape[:2], dtype=np.uint8)
    if not mask_path.exists():
        return mask

    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return mask

    mask = (mask_gray > MASK_THRESHOLD).astype(np.uint8)
    if mask.shape != target_shape[:2]:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def main():
    pairs = collect_image_mask_pairs(DATASET_ROOT)
    print(f"総フレーム数: {len(pairs)} (動画={len(set(p[2] for p in pairs))})")

    positive_mask_files = 0
    positive_pixels = 0
    processed = 0

    for img_path, mask_path, video_name in pairs:
        if processed >= MAX_IMAGES:
            break

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"画像読み込み失敗: {img_path}")
            continue

        mask = load_binary_mask(mask_path, image.shape)
        if mask_path.exists():
            positive_mask_files += 1
        if np.any(mask):
            positive_pixels += 1

        overlay = image.copy()
        overlay[mask == 1] = CLASS_COLOR_BGR

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, CLASS_COLOR_BGR, 2)

        blended = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)

        cv2.rectangle(blended, (10, 10), (30, 30), CLASS_COLOR_BGR, -1)
        cv2.putText(blended, CLASS_NAME, (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        output_path = output_dir / f"{video_name}_{img_path.name}"
        cv2.imwrite(str(output_path), blended)

        processed += 1
        if processed % 50 == 0:
            print(f"  {processed} 枚処理しました...")

    print("\n" + "=" * 60)
    print(f"保存先: {output_dir.resolve()}")
    print(f"処理枚数: {processed}")
    print(f"マスクファイルあり: {positive_mask_files} / {len(pairs)}")
    print(f"マスクが1以上の画像: {positive_pixels} / {len(pairs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
