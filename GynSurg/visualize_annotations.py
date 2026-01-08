import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
from collections import defaultdict

# Load COCO format annotation file
with open('Instrument_Anatomy_Original_Dataset/instruments.json', 'r') as f:
    coco_data = json.load(f)

# Target class IDs and their colors (BGR format for OpenCV)
TARGET_CLASSES = {
    14: ('in-cannula', (255, 255, 0))   # Cyan
}

# Create category lookup
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
print(f"Total categories: {len(categories)}")
print(f"Target classes: {TARGET_CLASSES}")

# Create image lookup
images_dict = {img['id']: img for img in coco_data['images']}
print(f"Total images: {len(images_dict)}")

# Group annotations by image
annotations_by_image = defaultdict(list)
for ann in coco_data['annotations']:
    if ann['category_id'] in TARGET_CLASSES:
        annotations_by_image[ann['image_id']].append(ann)

print(f"Images with target class annotations: {len(annotations_by_image)}")

# Count images per class
class_image_counts = defaultdict(set)
for image_id, annotations in annotations_by_image.items():
    for ann in annotations:
        cat_id = ann['category_id']
        class_image_counts[cat_id].add(image_id)

print("\nImages per class:")
for cat_id, (class_name, _) in TARGET_CLASSES.items():
    count = len(class_image_counts[cat_id])
    print(f"  {class_name}: {count} images")

# Create output directory
base_output_dir = Path('../vis/GynSurg')

# Create class-specific directories
class_output_dirs = {}
for cat_id, (class_name, _) in TARGET_CLASSES.items():
    class_dir = base_output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    class_output_dirs[cat_id] = class_dir

# Process each class separately
max_images_per_class = 300  # Limit number of images per class

for cat_id, (class_name, color) in TARGET_CLASSES.items():
    print(f"\n{'='*60}")
    print(f"Processing class: {class_name}")
    print(f"{'='*60}")

    processed_count = 0

    # Get images containing this class
    target_image_ids = class_image_counts[cat_id]

    for image_id in sorted(target_image_ids):
        if processed_count >= max_images_per_class:
            break

        # Get image info
        img_info = images_dict.get(image_id)
        if not img_info:
            continue

        # Construct full image path
        img_path = Path('Instrument_Anatomy_Original_Dataset') / img_info['path']
        seg_path = str(img_path).replace('insseg', 'auxtool_mask').replace('.png', '_mask.png')

        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        # Load image
        print(seg_path)
        image = cv2.imread(str(img_path))
        seg_image = cv2.imread(seg_path)
        if image is None:
            print(f"Failed to load: {img_path}")
            continue
        seg_file = str(img_info['path']).replace('insseg', 'past_annt').replace('/1.mp4_/', '_').replace('/2.mp4_/', '_').replace('/3.mp4_/', '_').replace('/4.mp4_/', '_').replace('/5.mp4_/', '_').replace('/6.mp4_/', '_').replace('/7.mp4_/', '_').replace('/8.mp4_/', '_').replace('/9.mp4_/', '_').replace('/0.mp4_/', '_')
        out_file = str(img_info['path']).replace('insseg', 'need_annt').replace('/1.mp4_/', '_').replace('/2.mp4_/', '_').replace('/3.mp4_/', '_').replace('/4.mp4_/', '_').replace('/5.mp4_/', '_').replace('/6.mp4_/', '_').replace('/7.mp4_/', '_').replace('/8.mp4_/', '_').replace('/9.mp4_/', '_').replace('/0.mp4_/', '_')
        #print(out_file)
        os.makedirs(Path(out_file).parent, exist_ok=True)
        os.makedirs(Path(seg_file).parent, exist_ok=True)
        cv2.imwrite(out_file, image)
        cv2.imwrite(seg_file, seg_image)
        # Create overlay image
        overlay = image.copy()

        # Get all annotations for this image (not just target classes)
        all_image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # For in-cannula, create semantic segmentation mask and exclude overlapping areas
        if class_name == 'in-cannula':
            # Create semantic segmentation masks
            h, w = image.shape[:2]
            in_cannula_mask = np.zeros((h, w), dtype=np.uint8)
            other_classes_mask = np.zeros((h, w), dtype=np.uint8)

            # First, create in-cannula mask
            for ann in all_image_annotations:
                if ann['category_id'] != cat_id:
                    continue

                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(in_cannula_mask, [points], 1)

            # Then, create mask for all other classes (to subtract from in-cannula)
            for ann in all_image_annotations:
                if ann['category_id'] == cat_id:
                    continue

                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(other_classes_mask, [points], 1)

            # Subtract other classes from in-cannula (keep only non-overlapping pixels)
            in_cannula_only_mask = np.logical_and(in_cannula_mask == 1, other_classes_mask == 0).astype(np.uint8)

            # Draw only the non-overlapping in-cannula pixels
            overlay[in_cannula_only_mask == 1] = color

            # Draw outline for in-cannula
            contours, _ = cv2.findContours(in_cannula_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw bbox for in-cannula annotations
            for ann in all_image_annotations:
                if ann['category_id'] != cat_id:
                    continue
                if 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        else:
            # For other classes, draw only that class
            for ann in all_image_annotations:
                if ann['category_id'] != cat_id:
                    continue

                # Handle segmentation (polygon format)
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        # Convert to polygon points
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)

                        # Draw filled polygon
                        cv2.fillPoly(overlay, [points], color)

                        # Draw polygon outline
                        cv2.polylines(overlay, [points], True, color, 2)

                # Draw bbox (always draw if available)
                if 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Blend original image with overlay
        alpha = 0.2
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        # Add legend - show only the current class
        #cv2.rectangle(blended, (10, 10), (30, 30), color, -1)
        #cv2.putText(blended, class_name, (40, 25),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save visualized image
        output_path = class_output_dirs[cat_id] / f"{img_info['file_name']}"
        cv2.imwrite(str(output_path), blended)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count} images...")

    print(f"  Total: {processed_count} images saved to {class_output_dirs[cat_id]}")

print(f"\n{'='*60}")
print(f"All done! Output saved to: {base_output_dir.absolute()}")
print(f"{'='*60}")
