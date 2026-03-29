import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def find_results_files():
    yolo_file = None
    rt_file = None
    
    print("Searching for your results files...")
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "results.csv":
                full_path = os.path.join(root, file)
                
                if "fire_detection_project" in root or "yolov8" in root:
                    yolo_file = full_path
                    print(f"✅ Found YOLOv8 results at: {yolo_file}")
                elif "rtdetr" in root:
                    rt_file = full_path
                    print(f"✅ Found RT-DETR results at: {rt_file}")
    
    return yolo_file, rt_file

def plot_comparison():
    yolo_csv, rt_csv = find_results_files()

    if not yolo_csv or not rt_csv:
        print("FAILED: Could not find both results files. Please ensure you haven't deleted them.")
        return

    df_yolo = pd.read_csv(yolo_csv)
    df_rt = pd.read_csv(rt_csv)

    df_yolo.columns = [c.strip() for c in df_yolo.columns]
    df_rt.columns = [c.strip() for c in df_rt.columns]
        
    y_prec = next((c for c in df_yolo.columns if 'precision' in c.lower()), None)
    y_rec = next((c for c in df_yolo.columns if 'recall' in c.lower()), None)
    y_map50 = next((c for c in df_yolo.columns if 'map50' in c.lower() and '95' not in c.lower()), None)
    
    r_prec = next((c for c in df_rt.columns if 'precision' in c.lower()), None)
    r_rec = next((c for c in df_rt.columns if 'recall' in c.lower()), None)
    r_map50 = next((c for c in df_rt.columns if 'map50' in c.lower() and '95' not in c.lower()), None)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('YOLOv8 vs RT-DETR: Performance Comparison', fontsize=16, fontweight='bold')

    axs[0].plot(df_yolo[y_prec], label='YOLOv8', color='blue')
    axs[0].plot(df_rt[r_prec], label='RT-DETR', color='orange')
    axs[0].set_title('Precision')
    axs[0].legend()

    axs[1].plot(df_yolo[y_rec], label='YOLOv8', color='blue')
    axs[1].plot(df_rt[r_rec], label='RT-DETR', color='orange')
    axs[1].set_title('Recall')
    axs[1].legend()

    axs[2].plot(df_yolo[y_map50], label='YOLOv8', color='blue')
    axs[2].plot(df_rt[r_map50], label='RT-DETR', color='orange')
    axs[2].set_title('mAP@50')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('Comparison_Curves.png')
    print("Saved Line Graphs to 'Comparison_Curves.png'")

    metrics = ['Precision', 'Recall', 'mAP50']
    yolo_vals = [df_yolo[y_prec].max(), df_yolo[y_rec].max(), df_yolo[y_map50].max()]
    rt_vals = [df_rt[r_prec].max(), df_rt[r_rec].max(), df_rt[r_map50].max()]

    x = np.arange(len(metrics))
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x - width/2, yolo_vals, width, label='YOLOv8', color='blue')
    ax2.bar(x + width/2, rt_vals, width, label='RT-DETR', color='orange')

    ax2.set_ylabel('Scores')
    ax2.set_title('Peak Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()

    plt.savefig('Comparison_Bar_Chart.png')
    print("Saved Bar Graph to 'Comparison_Bar_Chart.png'")

if __name__ == '__main__':
    plot_comparison()