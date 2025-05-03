import time
import torch
from ultralytics import YOLO

# Load your YOLO model (e.g., yolov8s.pt)
model = YOLO("yolov8s.pt")  # Replace with the appropriate model path

# Load the dataset (replace with your dataset)
dataset = "/workspace/datasets/dataset.yaml"  # Replace with the correct path to your dataset

def test_img_size(model, dataset, img_size):
    start_time = time.time()

    # Modify the image size for the model and dataset
    model.imgsz = img_size

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Reduce batch size if necessary to avoid memory overload
    batch_size = 8  # Change this as needed

    # Run the validation step (skip full training for faster results)
    results = model.val(data=dataset, imgsz=img_size, batch=batch_size)  # Replace batch_size with batch

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Get the mAP scores from validation results
    mAP50 = results['metrics']['mAP_0.5']
    mAP50_95 = results['metrics']['mAP_0.5:0.95']

    return {'img_size': img_size, 'mAP50': mAP50, 'mAP50-95': mAP50_95, 'time_taken': elapsed_time}

# List of image sizes to test
img_sizes = [640, 768, 1024, 1280, 1600, 2048]

# Run the tests sequentially
results = []
for img_size in img_sizes:
    result = test_img_size(model, dataset, img_size)
    results.append(result)

# Print results
for result in results:
    print(f"Image Size: {result['img_size']} | mAP50: {result['mAP50']:.4f} | mAP50-95: {result['mAP50-95']:.4f} | Time: {result['time_taken']:.2f} seconds")
