import torch
import cv2
import numpy as np
import platform
import pathlib
from pathlib import Path

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

model_path = pathlib.Path("C:/Users/Rashmin/Downloads/exam_proctoring new/exp/weights/best.pt").resolve()
model_path = str(model_path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

input_folder = Path('C:/Users/Rashmin/Downloads/exam_proctoring new/exp/images')
output_folder = Path('C:/Users/Rashmin/Downloads/exam_proctoring new/exp/Result')

output_folder.mkdir(parents=True, exist_ok=True)

image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.png'))

for image_path in image_files:
    image = cv2.imread(str(image_path))
    if image is None:
        continue 
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)

    for *box, conf, cls in results.xyxy[0]:  
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls)]
        confidence = conf.item()

        label = f'{class_name} {confidence:.2f}'
        color = (0, 255, 0)  

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = output_folder / image_path.name
    cv2.imwrite(str(output_path), image)

print("Detection complete. Images saved with detected objects.")
