import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def polygon_to_bbox(polygon):
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    return xmin, ymin, xmax, ymax


def run_inference(models, images_path, labels_path, img_size):
    models = {name: {'model': YOLO(info['path']), 'color': info['color']} for name, info in models.items()}

    results_path = os.path.join('outputs', "results")
    os.makedirs(results_path, exist_ok=True)

    for image_file in os.listdir(images_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(images_path, image_file)
            label_path = os.path.join(labels_path, image_file.replace('.jpg', '.txt'))

            image = cv2.imread(image_path)
            height, width, _ = image.shape

            actual_boxes = []
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    polygon = list(map(float, parts[1:]))
                    xmin, ymin, xmax, ymax = polygon_to_bbox(polygon)
                    xmin, ymin, xmax, ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(
                        ymax * height)
                    actual_boxes.append((class_id, xmin, ymin, xmax, ymax))

            results = {name: model_info['model'].predict(source=image_path, conf=0.75, imgsz=img_size, show=False)
                       for name, model_info in models.items()}

            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw actual annotations
            for class_id, xmin, ymin, xmax, ymax in actual_boxes:
                ax.add_patch(
                    Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='none'))

            predictions = {name: [] for name in models.keys()}

            for name, model_info in models.items():
                for box in results[name][0].boxes:
                    pred_class_id = int(box.cls)
                    conf = float(box.conf)
                    predictions[name].append((pred_class_id, conf, box))
                    bbox = box.xyxy.cpu().numpy().astype(int)[0]
                    xmin, ymin, xmax, ymax = bbox
                    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                           edgecolor=tuple([c / 255 for c in model_info['color']]), facecolor='none'))

            for name in predictions:
                predictions[name].sort(key=lambda x: x[1], reverse=True)

            legend_elements = [Line2D([0], [0], color='red', lw=2, label=f"TRUTH: {actual_boxes[0][0]}")]
            for name, preds in predictions.items():
                for pred_class_id, conf, _ in preds:
                    legend_elements.append(
                        Line2D([0], [0], color=tuple([c / 255 for c in models[name]['color']]), lw=2, linestyle='-',
                               label=f"{name.upper()} PRED: {pred_class_id} ({conf:.2f})"))

            ax.legend(handles=legend_elements, loc='upper right', fontsize=28)

            output_path = os.path.join(results_path, image_file)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved annotated image: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on validation images")
    parser.add_argument('--model_paths', nargs='+', help="Paths to the trained model files", required=True)
    parser.add_argument('--images_path', help="Path to the validation images", required=True)
    parser.add_argument('--labels_path', help="Path to the validation labels", required=True)
    parser.add_argument('--img_size', type=int, default=512, help="Image size for inference (default: 512)")
    args = parser.parse_args()

    # Define colors for models and ground truth
    colors = [
        (255, 0, 0),  # Red for ground truth
        (255, 165, 0),  # Orange for basic model
        (255, 255, 0),  # Yellow for SA model
        (0, 255, 0),  # Green for CA model
        (0, 0, 255),  # Blue for ResCBAM model
        (128, 0, 128)  # Purple for SCSA model
    ]

    if len(args.model_paths) > len(colors) - 1:
        print("Too many models provided, maximum supported is 5 models.")
        exit()

    model_info = [{'path': args.model_paths[i], 'color': colors[i + 1]} for i in range(len(args.model_paths))]
    models = {f'model_{i + 1}': info for i, info in enumerate(model_info)}

    run_inference(models, args.images_path, args.labels_path, args.img_size)
