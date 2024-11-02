from tkinter import Image
from ultralytics import YOLO


model = YOLO('best.pt')
results = model(source=3, show=True, conf= 0.4, save=False)