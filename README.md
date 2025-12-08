# Road Users Detection with YOLO

This project demonstrates how to prepare a custom dataset, convert XML annotations to YOLO format, organize training data, and train a YOLO model for object detection.
The dataset contains street images with annotated objects such as vehicles and pedestrians.

## Dependencies

torch
torchvision
Pillow
xml.etree
pandas
ultralytics
yaml
tqdm


## Data Structure

archive/
├── images/          # All raw images (.jpg/.png)
└── annotations/     # Annotation files in Pascal VOC (.xml)

## Road user categories

classes = ["passenger_car", "lorry", "utility_vehicle", "bus", "tram", "pedestrian"]

## Annotation Parsing

The script extracts:
Object name
Bounding box coordinates (xmin, ymin, xmax, ymax)
Image size

### Then converts bounding boxes into YOLO format:
x_center, y_center, width, height  (normalized between 0–1)
example: 
[4 0.8293 0.4500 0.3398 0.4972] = [class_id x_center y_center width height]

## Data Conversion and Split

Steps performed:

Read images and XML files

Convert annotations into YOLO text files (.txt)

Convert images into .jpg format

Randomly split dataset into train (80%) and val (20%)

### Save processed data into:
data_yolo/
├── images/
│   ├── train
│   └── val
└── labels/
    ├── train
    └── val


path: <absolute_path_to_data_yolo>
train: images/train
val: images/val
names: [passenger_car, lorry, utility_vehicle, bus, tram, pedestrian]
nc: 6

## Model tranning
model = YOLO("yolo11s.pt")
model.train(data="data.yaml", epochs=..., imgsz=...)

## Model Evaluation / Prediction

pred_results = model.predict(img_path, conf=0.5)

| Field | Description               |
| ----- | ------------------------- |
| cls   | predicted class index     |
| conf  | confidence score          |
| xyxy  | bounding box coordinates  |
| xyxyn | normalized bounding boxes |

## Exporting Predictions

Predictions are stored into a Pandas DataFrame format:

ID	image_id	confidence	class_name	x_min	y_min	x_max	y_max

This makes the result easy to save to CSV, Excel, or database.




