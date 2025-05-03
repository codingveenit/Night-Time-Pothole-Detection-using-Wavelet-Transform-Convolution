from ultralytics import YOLO
import cv2
import os

if __name__ == '__main__':
    # model = YOLO('runs/wt-yolov8/weights/best.pt')
    model = YOLO('runs/wt-yolov843/weights/best.pt')

    test_images = 'datasets/images/test/'
    output_dir = 'predictions/'
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(source=test_images, conf=0.25, save=False)

    for result in results:
        img = result.plot()
        filename = os.path.basename(result.path)
        cv2.imwrite(os.path.join(output_dir, filename), img)

    print(f"Predictions saved to {output_dir}")
