import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_density(pred, true_labels,path, file_name,  threshold):
    plt.close()
    pred_1 = [x if y > threshold else 0 for x, y in zip(pred[:, 1], true_labels)]
    pred_1 = [x for x in pred_1 if x != 0]

    pred_0 = [x if y < threshold else 0 for x, y in zip(pred[:, 1], true_labels)]
    pred_0 = [x for x in pred_0 if x != 0]

    # Create a density plot using seaborn
    sns.kdeplot(pred_0, fill=True)
    sns.kdeplot(pred_1, fill=True)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Density Plot of Predicted Probabilities')
    plt.axvline(x=threshold, color='r', linestyle='--')
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    plt.savefig(path  / file_name)
    plt.close()