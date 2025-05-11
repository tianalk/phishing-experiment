import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULTS_DIR = 'results'
SUMMARY_PATH = os.path.join(RESULTS_DIR, 'summary_report.json')
VIS_DATA_PATH = os.path.join(RESULTS_DIR, 'visualization_data.json')

with open(SUMMARY_PATH, 'r') as f:
    summary = json.load(f)

# Helper to plot confusion matrix
def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# Helper to plot bar metrics
def plot_bar_metrics(metrics, title, filename):
    classes = list(metrics.keys())
    precision = [metrics[c]['Precision'] for c in classes]
    recall = [metrics[c]['Recall'] for c in classes]
    f1 = [metrics[c]['F1-Score'] for c in classes]
    x = np.arange(len(classes))
    width = 0.25
    plt.figure(figsize=(7,5))
    plt.bar(x-width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x+width, f1, width, label='F1-Score')
    plt.xticks(x, classes)
    plt.ylim(0,1.1)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# Overall metrics
for key, label in [("Human_Generated_Metrics", "Human"), ("AI_Generated_Metrics", "AI")]:
    m = summary[key]
    plot_confusion_matrix(m['Confusion_Matrix'], m['Labels'], f'{label} Overall Confusion Matrix', f'{label.lower()}_confusion_matrix.png')
    plot_bar_metrics(m['Per_Class'], f'{label} Per-Class Metrics', f'{label.lower()}_per_class_metrics.png')
    # Breakdown for phishing types
    if 'Phishing_Type_Breakdown' in m:
        for ptype, b in m['Phishing_Type_Breakdown'].items():
            plot_confusion_matrix(b['Confusion_Matrix'], b['Labels'], f'{label} {ptype} Confusion Matrix', f'{label.lower()}_{ptype.replace(" ", "_")}_confusion_matrix.png')
            plot_bar_metrics(b['Per_Class'], f'{label} {ptype} Per-Class Metrics', f'{label.lower()}_{ptype.replace(" ", "_")}_per_class_metrics.png')

# Accuracy comparison plot
labels = ['Overall', 'AI-generated phishing', 'NK phishing']
human_acc = [summary['Human_Generated_Metrics']['Accuracy'],
             summary['Human_Generated_Metrics']['Phishing_Type_Breakdown']['ai-generated phishing']['Accuracy'],
             summary['Human_Generated_Metrics']['Phishing_Type_Breakdown']['phishing']['Accuracy']]
ai_acc = [summary['AI_Generated_Metrics']['Accuracy'],
          summary['AI_Generated_Metrics']['Phishing_Type_Breakdown']['ai-generated phishing']['Accuracy'],
          summary['AI_Generated_Metrics']['Phishing_Type_Breakdown']['phishing']['Accuracy']]
x = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(7,5))
plt.bar(x-width/2, human_acc, width, label='Human')
plt.bar(x+width/2, ai_acc, width, label='AI')
plt.xticks(x, labels)
plt.ylim(0,1.1)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_comparison.png'))
plt.close()

# Export JSON for Three.js
with open(VIS_DATA_PATH, 'w') as f:
    json.dump(summary, f, indent=2) 