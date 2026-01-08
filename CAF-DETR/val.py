import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model_path = 'runs/train/exp13/weights/best.pt'
    model = RTDETR(model_path)
    result = model.val(data='dataset/data.yaml',
                      split='test',
                      imgsz=640,
                      batch=4,
                      project='runs/val',
                      name='exp',
                      )
