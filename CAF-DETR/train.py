import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/CAF-DETR.yaml')
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=4,
                workers=4,
                device='0,1',
                project='runs/train',
                name='exp',
                )