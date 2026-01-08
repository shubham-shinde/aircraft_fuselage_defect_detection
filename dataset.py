from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG
model = YOLO('yolov8n.pt') 
from ultralytics.data.utils import check_det_dataset

from ultralytics.utils import DEFAULT_CFG

file = 'AFDET_dataset.yaml'
data = YAML.load(file)  # dictionary
data = check_det_dataset(file)

# if you use mode 'train' it'll apply unwanted augmentations
train_dataset = build_yolo_dataset(DEFAULT_CFG, data['train'], 1, data, mode='val')
val_dataset = build_yolo_dataset(DEFAULT_CFG, data['val'], 1, data, mode='val')