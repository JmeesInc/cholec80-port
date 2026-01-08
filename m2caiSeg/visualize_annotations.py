import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict

########################################################
# Trocars class ID: 8
# Trocars color (BGR): [170, 85, 85]
########################################################
class_info_dict = {}
class_info_dict[8] = {
    'name': 'trocars',
    'color_rgb': [170, 85, 85],
    'visualization_color_bgr': (255, 255, 0)
}
print(f"Total classes: {len(class_info_dict)}")
print(f"Classes: {[info['name'] for info in class_info_dict.values()]}")

# Data directories
data_root = Path(__file__).parent
trainval_image_dir = data_root / 'train' / 'images'
trainval_mask_dir = data_root / 'train' / 'groundtruth'
test_image_dir = data_root / 'test' / 'images'
test_mask_dir = data_root / 'test' / 'groundtruth'

# Collect all image-mask pairs
image_mask_pairs = []

# Process trainval set
if trainval_image_dir.exists() and trainval_mask_dir.exists():
    for img_file in sorted(trainval_image_dir.glob("*.jpg")) + sorted(trainval_image_dir.glob("*.png")):
        mask_name = img_file.stem + "_gt.png"
        mask_file = trainval_mask_dir / mask_name
        if mask_file.exists():
            image_mask_pairs.append((img_file, mask_file, 'trainval'))

# Process test set
if test_image_dir.exists() and test_mask_dir.exists():
    for img_file in sorted(test_image_dir.glob("*.jpg")) + sorted(test_image_dir.glob("*.png")):
        mask_name = img_file.stem + "_gt.png"
        mask_file = test_mask_dir / mask_name
        if mask_file.exists():
            image_mask_pairs.append((img_file, mask_file, 'test'))

print(f"\nTotal image-mask pairs found: {len(image_mask_pairs)}")

# Count images per class
class_image_counts = defaultdict(set)
for img_path, mask_path, split in image_mask_pairs:
    mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
    for class_id, class_info in class_info_dict.items():
        class_color_rgb = class_info['color_rgb']
        class_mask = np.all(mask_rgb == class_color_rgb, axis=2)
        if np.any(class_mask):
            class_image_counts[class_id].add((img_path, mask_path, split))

print("\nImages per class:")
for class_id in sorted(class_image_counts.keys()):
    class_name = class_info_dict[class_id]['name']
    count = len(class_image_counts[class_id])
    print(f"  {class_name}: {count} images")

# Create output directory
base_output_dir = Path('../vis/m2caiSeg')

# Create class-specific directories
class_output_dirs = {}
for class_id, class_info in class_info_dict.items():
    class_dir = base_output_dir / class_info['name']
    class_dir.mkdir(parents=True, exist_ok=True)
    class_output_dirs[class_id] = class_dir

# Process each class separately
max_images_per_class = 300  # Limit number of images per class

for class_id, class_info in sorted(class_info_dict.items()):
    class_name = class_info['name']
    class_color_rgb = class_info['color_rgb']
    visualization_color_bgr = class_info['visualization_color_bgr']
    
    print(f"\n{'='*60}")
    print(f"Processing class: {class_name} (ID: {class_id})")
    print(f"Original color (RGB): {class_color_rgb}")
    print(f"Visualization color (BGR, complementary): {visualization_color_bgr}")
    print(f"{'='*60}")

    processed_count = 0
    
    # Collect resolutions for statistics
    resolutions = []

    # Get images containing this class
    target_images = class_image_counts[class_id]

    for img_path, mask_path, split in sorted(target_images):
        if processed_count >= max_images_per_class:
            break

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load: {img_path}")
            continue
        
        out_file = str(img_path).replace('test/images/', 'need_annt/test/').replace('trainval/images/', 'need_annt/trainval/')
        os.makedirs(Path(out_file).parent, exist_ok=True)
        #print(out_file)
        cv2.imwrite(out_file, image)
        # Collect resolution for statistics
        height, width = image.shape[:2]
        resolutions.append((width, height))

        # Load mask
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        
        # Create class mask
        class_mask = np.all(mask_rgb == class_color_rgb, axis=2).astype(np.uint8)

        # Create mask for all other classes (to subtract from current class)
        other_classes_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for other_class_id, other_class_info in class_info_dict.items():
            if other_class_id == class_id:
                continue
            other_class_color_rgb = other_class_info['color_rgb']
            other_class_mask = np.all(mask_rgb == other_class_color_rgb, axis=2)
            other_classes_mask = np.logical_or(other_classes_mask, other_class_mask).astype(np.uint8)

        # Subtract other classes from current class (keep only non-overlapping pixels)
        class_only_mask = np.logical_and(class_mask == 1, other_classes_mask == 0).astype(np.uint8)

        # Create overlay image
        overlay = image.copy()

        # Draw only the non-overlapping class pixels
        overlay[class_only_mask == 1] = visualization_color_bgr

        # Draw outline for class
        contours, _ = cv2.findContours(class_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, visualization_color_bgr, 2)

        # Blend original image with overlay
        alpha = 0.2
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        # Add legend
        #cv2.rectangle(blended, (10, 10), (30, 30), visualization_color_bgr, -1)
        #cv2.putText(blended, class_name, (40, 25),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save visualized image
        output_filename = f"{split}_{img_path.stem}.png"
        output_path = class_output_dirs[class_id] / output_filename
        cv2.imwrite(str(output_path), blended)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count} images...")

    print(f"  Total: {processed_count} images saved to {class_output_dirs[class_id]}")
    
    # Print resolution statistics
    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        min_width, max_width = min(widths), max(widths)
        min_height, max_height = min(heights), max(heights)
        print(f"  Resolution statistics:")
        print(f"    Width:  min={min_width}, max={max_width}")
        print(f"    Height: min={min_height}, max={max_height}")
        print(f"    Min resolution: {min_width}x{min_height}")
        print(f"    Max resolution: {max_width}x{max_height}")

print(f"\n{'='*60}")
print(f"All done! Output saved to: {base_output_dir.absolute()}")
print(f"{'='*60}")
