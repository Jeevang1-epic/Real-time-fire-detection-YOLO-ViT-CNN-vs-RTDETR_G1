from ultralytics import YOLO

def run_inference():
    model = YOLO('runs/detect/fire_detection_project/yolov8_training_run/weights/best.pt')
    
    model.predict(
        source='building fire.mp4',
        save=True,
        project='fire_visuals',
        name='video_output',
        conf=0.4
    )

if __name__ == '__main__':
    run_inference()