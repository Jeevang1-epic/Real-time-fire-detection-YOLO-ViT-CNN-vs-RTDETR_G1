from ultralytics import YOLO

def main():
    model = YOLO('best.pt')
    
    print("Generating Confusion Matrix, F1 Curve, and PR Curve...")
    metrics = model.val(
        data='data.yaml', 
        plots=True,       
        save_json=True,
        project='fire_visuals',
        name='final_evaluation'
    )
    print("Done! Check the 'fire_visuals' folder.")

if __name__ == '__main__':
    main()