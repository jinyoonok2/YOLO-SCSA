from ultralytics import YOLO
import os

def get_model_path(model_type, size):
    base_path = r".\ultralytics\cfg\models\v8"
    model_paths = {
        'basic': f"yolov8{size}.yaml",
        'sa': f"yolov8{size}_SA.yaml",
        'ca': f"yolov8{size}_CA.yaml",
        'rescbam': f"yolov8{size}_SCSA_CWM.yaml",
        'scsa': f"yolov8{size}_SCSA_CWM.yaml"
    }
    model_file = model_paths.get(model_type.lower())
    if not model_file:
        raise ValueError(f"Invalid model type: {model_type}")
    return os.path.join(base_path, model_file)

def get_experiment_name(model_type, size):
    name_map = {
        'basic': 'Basic',
        'sa': 'SA',
        'ca': 'CA',
        'rescbam': 'ResCBAM',
        'scsa': 'SCSA'
    }
    model_name = name_map.get(model_type.lower())
    if not model_name:
        raise ValueError(f"Invalid model type: {model_type}")
    return f"{model_name}({size})"

def train(model_type, data_path, img_size, size):
    model_path = get_model_path(model_type, size)
    experiment_name = get_experiment_name(model_type, size)
    os.makedirs('./logs', exist_ok=True)
    model = YOLO(model_path)
    model.train(data=data_path, imgsz=img_size, epochs=100, name=experiment_name, project='./logs')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model")
    parser.add_argument('model_type', help="Model type: basic, sa, ca, rescbam, scsa")
    parser.add_argument('data_path', help="Path to the dataset YAML file")
    parser.add_argument('--img_size', type=int, default=512, help="Image size for training (default: 512)")
    parser.add_argument('--size', default='s', choices=['n', 's', 'm', 'l', 'x'], help="Model size: n, s, m, l, x (default: s)")
    args = parser.parse_args()
    train(args.model_type, args.data_path, args.img_size, args.size)
