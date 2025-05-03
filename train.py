from ultralytics import YOLO
from model.wtconv import WTConv
import torch

def modify_model():
    model = YOLO('yolov8s.pt')
    
    # Get the details of the target layer
    target_layer = model.model.model[1]
    in_channels = target_layer.conv.in_channels
    out_channels = target_layer.conv.out_channels
    
    print(f"Replacing layer with in_channels={in_channels}, out_channels={out_channels}")
    
    # Replace with WTConv using the same dimensions
    model.model.model[1] = WTConv(in_channels=in_channels//4, out_channels=out_channels, wavelet='db1')
    
    return model

if __name__ == '__main__':
    model = modify_model()
    
    model.train(
        data='/workspace/datasets/dataset.yaml',
        epochs=70,
        imgsz=512,
        # imgsz=100,
        batch=32,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.05,
        project='runs',
        name='wt-yolov8',
        workers=4
    )
