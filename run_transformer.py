from ultralytics import RTDETR
import os

def main():
    print("Starting RT-DETR Transformer Pipeline (Memory Safe Mode)...")
    
    model = RTDETR('rtdetr-l.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=200,
        imgsz=640,
        device='0',
        batch=8,           
        workers=4,         
        cache=True,        
        project='runs/detect',
        name='rtdetr_run_fast', 
        plots=True
    )
    
    save_dir = str(results.save_dir)
    print(f"\nTransformer Training complete. Data safe in: {save_dir}")
    
    print("\nGenerating F1 Scores and Confusion Matrices...")
    best_weights = os.path.join(save_dir, 'weights', 'best.pt')
    
    if os.path.exists(best_weights):
        best_model = RTDETR(best_weights)
        best_model.val(data='data.yaml', plots=True, project=save_dir, name='eval_matrices')
        
        print("\nProcessing drone video with RT-DETR...")
        best_model.predict(
            source='building fire.mp4',
            save=True,
            project=save_dir,
            name='video_output',
            conf=0.4
        )
        print("\nPipeline Complete! Visuals and video are ready.")
    else:
        print("Error: Could not find best.pt weights.")

if __name__ == '__main__':
    main()