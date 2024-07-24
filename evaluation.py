from ultralytics import YOLO
import os

def evaluate(model_path, data_path, img_size):
    if not os.path.isfile(model_path):
        print(f"Provided model path does not exist: {model_path}")
        return

    model = YOLO(model_path)
    model.val(data=data_path, imgsz=img_size, project='./logs', device=0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model")
    parser.add_argument('model_path', help="Path to the trained model file")
    parser.add_argument('data_path', help="Path to the dataset YAML file")
    parser.add_argument('--img_size', type=int, default=512, help="Image size for evaluation (default: 512)")
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path, args.img_size)
