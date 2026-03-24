import pandas as pd
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

def plot_custom_metrics(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    train_box = next((c for c in df.columns if 'train/box_loss' in c), None)
    val_box = next((c for c in df.columns if 'val/box_loss' in c), None)
    train_cls = next((c for c in df.columns if 'train/cls_loss' in c), None)
    val_cls = next((c for c in df.columns if 'val/cls_loss' in c), None)
    precision = next((c for c in df.columns if 'precision' in c or 'Precision' in c), None)
    recall = next((c for c in df.columns if 'recall' in c or 'Recall' in c), None)
    map50 = next((c for c in df.columns if 'mAP50' in c and '95' not in c), None)
    map95 = next((c for c in df.columns if 'mAP50-95' in c), None)

    epochs = df['epoch']

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fire Detection Training Metrics', fontsize=16, fontweight='bold')

    axs[0, 0].plot(epochs, df[train_box], label='Train Box Loss', color='blue', linewidth=2)
    axs[0, 0].plot(epochs, df[val_box], label='Val Box Loss', color='cyan', linewidth=2)
    axs[0, 0].set_title('Bounding Box Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    axs[0, 1].plot(epochs, df[train_cls], label='Train Class Loss', color='red', linewidth=2)
    axs[0, 1].plot(epochs, df[val_cls], label='Val Class Loss', color='orange', linewidth=2)
    axs[0, 1].set_title('Classification Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    axs[1, 0].plot(epochs, df[precision], label='Precision', color='purple', linewidth=2)
    axs[1, 0].plot(epochs, df[recall], label='Recall', color='magenta', linewidth=2)
    axs[1, 0].set_title('Precision & Recall')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    axs[1, 1].plot(epochs, df[map50], label='mAP@50', color='green', linewidth=2)
    axs[1, 1].plot(epochs, df[map95], label='mAP@50-95', color='lightgreen', linewidth=2)
    axs[1, 1].set_title('Mean Average Precision')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('custom_training_metrics.png', dpi=300)
    print("Saved custom portfolio metrics to 'custom_training_metrics.png'")

def main():
    print("Starting Unified 200-Epoch Training Run...")
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=200,
        imgsz=640,
        device='0',
        batch=16,
        amp=True,
        plots=True 
    )
    
    save_dir = str(results.save_dir)
    print(f"\n Training complete. Outputs saved to: {save_dir}")
    
    csv_path = os.path.join(save_dir, 'results.csv')
    if os.path.exists(csv_path):
        plot_custom_metrics(csv_path)
        
    print("\nForcing high-quality validation visuals (F1, Confusion Matrix)...")
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        best_model.val(data='data.yaml', plots=True)
        print("ALL DONE! Your model and visuals are 100% ready.")

if __name__ == '__main__':
    main()