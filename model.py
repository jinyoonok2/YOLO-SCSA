import argparse
from train import train
from evaluation import evaluate
from inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Skin Cancer Detection Tasks")
    parser.add_argument('task', choices=['train', 'evaluate', 'inference'], help="Task to perform: train, evaluate, inference")
    parser.add_argument('--model_type', help="Model type for training: basic, sa, ca, rescbam, scsa")
    parser.add_argument('--model_paths', nargs='+', help="Paths to the trained model files for inference")
    parser.add_argument('--model_path', help="Path to the trained model file for evaluation")
    parser.add_argument('--data_path', help="Path to the data file for training or evaluation")
    parser.add_argument('--images_path', help="Path to the validation images for inference")
    parser.add_argument('--labels_path', help="Path to the validation labels for inference")
    parser.add_argument('--img_size', type=int, default=512, help="Image size for training, evaluation, or inference (default: 512)")
    parser.add_argument('--size', default='s', choices=['n', 's', 'm', 'l', 'x'], help="Model size: n, s, m, l, x (default: s)")
    args = parser.parse_args()

    if args.task == 'train':
        if not args.model_type or not args.data_path:
            print("Model type and data path are required for training")
            return
        train(args.model_type, args.data_path, args.img_size, args.size)
    elif args.task == 'evaluate':
        if not args.model_path or not args.data_path:
            print("Model path and data path are required for evaluation")
            return
        evaluate(args.model_path, args.data_path, args.img_size)
    elif args.task == 'inference':
        if not args.model_paths or not args.images_path or not args.labels_path:
            print("Model paths, images path, and labels path are required for inference")
            return
        # Define colors for models and ground truth
        colors = [
            (255, 0, 0),     # Red for ground truth
            (255, 165, 0),   # Orange for basic model
            (255, 255, 0),   # Yellow for SA model
            (0, 255, 0),     # Green for CA model
            (0, 0, 255),     # Blue for ResCBAM model
            (128, 0, 128)    # Purple for SCSA model
        ]

        if len(args.model_paths) > len(colors) - 1:
            print("Too many models provided, maximum supported is 5 models.")
            return

        model_info = [{'path': args.model_paths[i], 'color': colors[i + 1]} for i in range(len(args.model_paths))]
        models = {f'model_{i+1}': info for i, info in enumerate(model_info)}
        run_inference(models, args.images_path, args.labels_path, args.img_size)

if __name__ == '__main__':
    main()

# python model.py train --model_type scsa --data_path "./datasets/data_local.yaml" --img_size 512 --size m
# python model.py evaluate --model_path "./logs/SCSA(m)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
# python model.py inference --model_paths "./logs/Basic(s)/weights/best.pt" "./logs/SCSA(m)/weights/best.pt" "./logs/ResCBAM(m)/weights/best.pt" --images_path "./datasets/valid/images" --labels_path "./datasets/valid/labels" --img_size 512