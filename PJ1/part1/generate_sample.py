import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os


def generate_sin_samples(filename):
    x_min, x_max = -1 * np.pi, np.pi
    n_samples = 5000
    x = np.random.uniform(x_min, x_max, size=n_samples)
    y_true = np.sin(x)
    
    # add noise
    noise_scale = 0.1
    y_sample = y_true + np.random.normal(scale=noise_scale, size=n_samples)
    # avg_error = np.square(y_sample - y_true).mean()

    f = open(filename, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["x", "y"])
    for i in range(n_samples):
        csv_writer.writerow([str(x[i]), str(y_sample[i])])
    f.close()


def read_file(filename: str):
    data = pd.read_csv(filename, usecols=[0])
    x = np.array(data.stack()).reshape(-1, 1)
    label = pd.read_csv(filename, usecols=[1])
    y = np.array(label.stack()).reshape(-1, 1)
    return x, y


if __name__ == "__main__":
    if not os.path.exists("test.csv") : generate_sin_samples("test.csv")
    x, y = read_file("test.csv")
    y_sin = np.sin(x)
    avg_error = np.square(y_sin - y).mean()
    print("average error:", avg_error)

    plt.figure()
    plt.scatter(x, y, s=5, label="Sample Data")
    plt.scatter(x, y_sin, s=5, label="True Data")
    plt.legend()
    plt.show()
