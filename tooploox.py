import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv_data(input_csv_file):
    with open(input_csv_file, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        views_list = np.asarray(list(reader))
        return np.asarray(views_list[:, 1:], dtype=int)


def plot_histogram_view_vs_videos(data, title):
    plt.hist(data)
    plt.title(title)
    plt.xlabel("Number of views")
    plt.ylabel("Number of videos")
    plt.show()


def compute_mrse(predictor, input_data, ground_truth):

    return sum((predictor.predict(input_data)/ground_truth - 1)**2)/len(ground_truth)
