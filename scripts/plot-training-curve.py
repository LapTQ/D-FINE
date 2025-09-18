import json
import matplotlib.pyplot as plt
import os

def plot_metrics(log_files_with_names, output_dir):
    # Initialize dictionaries to store the data for each model
    models_data = {}

    # Read each log file
    for log_file, model_name in log_files_with_names:
        models_data[model_name] = {'epochs': [], 'map50_95': [], 'map50': []}

        with open(log_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                models_data[model_name]['epochs'].append(data['epoch'])
                models_data[model_name]['map50_95'].append(data['test_coco_eval_bbox'][0])
                models_data[model_name]['map50'].append(data['test_coco_eval_bbox'][1])

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot mAP50:95 for all models
    plt.figure(figsize=(10, 5))
    for model_name, data in models_data.items():
        plt.plot(data['epochs'], data['map50_95'], label=f'{model_name}')

    plt.xlabel('Epoch')
    plt.ylabel('mAP50:95')
    plt.title('mAP50:95')
    plt.legend()

    # Save the mAP50:95 figure
    map50_95_path = os.path.join(output_dir, 'map50_95_plot.png')
    plt.savefig(map50_95_path)
    plt.close()

    # Plot mAP50 for all models
    plt.figure(figsize=(10, 5))
    for model_name, data in models_data.items():
        plt.plot(data['epochs'], data['map50'], label=f'{model_name}')

    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.title('mAP50')
    plt.legend()

    # Save the mAP50 figure
    map50_path = os.path.join(output_dir, 'map50_plot.png')
    plt.savefig(map50_path)
    plt.close()

    print(f"Figures saved to {output_dir}")


# Example usage
log_files = [
    ("/home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_s/train/log.txt", "D-FINEs 960"),
]
output_dir = "/home/laptq/laptq-prj-46/outputs/plot-curve-dfine"
plot_metrics(log_files, output_dir)