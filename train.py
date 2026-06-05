import torch
from ultralytics import YOLO
import warnings

def main():
    model = YOLO("ultralytics/cfg/models/11/MEAQ-YOLO.yaml")


    model.train(
        data="/data.yaml",
        project="/runs/train",
        name="MEAQ-YOLO",
        epochs=200,
        imgsz=640,
        batch=16,
        workers=6,
        seed=42,
        deterministic=True,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )

    model.val()
if __name__ == "__main__":
    main()
