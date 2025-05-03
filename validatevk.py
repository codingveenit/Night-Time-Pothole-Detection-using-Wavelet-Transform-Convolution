from ultralytics import YOLO
import cv2
import os

if __name__ == '__main__':
    # Load the trained YOLO model
    model = YOLO('runs/wt-yolov843/weights/best.pt')  # Adjust the path if needed

    # Directory for test images and saving predictions
    test_images = 'datasets/images/test/'
    output_dir = 'predictionsvk/'
    os.makedirs(output_dir, exist_ok=True)

    # Run predictions on test dataset
    results = model.predict(source=test_images, conf=0.25, save=False)

    # Save predictions with bounding boxes drawn
    for result in results:
        img = result.plot()
        filename = os.path.basename(result.path)
        cv2.imwrite(os.path.join(output_dir, filename), img)

    print(f"\n✅ Predictions saved to: {output_dir}")

    # Evaluate model performance
    eval_results = model.val(data='/workspace/datasets/converted.yaml', imgsz=640, conf=0.25)

    # Extract evaluation metrics
    mAP_0_5_to_0_95 = eval_results.box.map
    all_maps = eval_results.box.maps

    # Print results
    print("\n📊 Evaluation Results:")
    print(f"mAP@[0.5:0.95]: {mAP_0_5_to_0_95:.4f}")
    print(f"AP@0.5: {all_maps[0]:.4f}" if len(all_maps) > 0 else "AP@0.5: Not available")
    print(f"AP@0.75: {all_maps[5]:.4f}" if len(all_maps) > 5 else "AP@0.75: Not available")

    # Save results to file
    with open("evaluation_results.txt", "w") as f:
        f.write(f"mAP@[0.5:0.95]: {mAP_0_5_to_0_95:.4f}\n")
        if len(all_maps) > 0:
            f.write(f"AP@0.5: {all_maps[0]:.4f}\n")
        else:
            f.write("AP@0.5: Not available\n")
        if len(all_maps) > 5:
            f.write(f"AP@0.75: {all_maps[5]:.4f}\n")
        else:
            f.write("AP@0.75: Not available\n")

    print("\n📁 Evaluation results saved to evaluation_results.txt")
