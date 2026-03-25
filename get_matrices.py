from ultralytics import YOLO
import os

def main():
    model_path = 'runs/detect/fire_detection_project/yolov8_training_run/weights/best.pt'
    
    if os.path.exists(model_path):
        model = YOLO(model_path)
        model.val(data='data.yaml', plots=True, project='fire_visuals', name='final_eval')
    else:
        print("Model weights not found. Check the path.")

if __name__ == '__main__':
    main()