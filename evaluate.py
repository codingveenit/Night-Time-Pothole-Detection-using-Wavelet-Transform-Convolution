from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/wt-yolov8/weights/best.pt')
    metrics = model.val(data='datasets/dataset.yaml', split='test')

    print(f"mAP: {metrics.box.map}")
    print(f"AP@0.5: {metrics.box.map50}")
    print(f"AP@0.75: {metrics.box.map75}")

    metrics.plot_pr_curve()
