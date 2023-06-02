import numpy as np


def calculate_mse(true_labels_file, predicted_labels_file, threshold):
    true_labels, predicted_labels = [], []

    # Read true labels from true_labels_file
    with open(true_labels_file, "r") as f:
        for line in f.readlines():
            if line.strip() == "NaN":
                true_labels.append(float("inf"))
                continue

            true_labels.append(round(float(line.strip())))

    # Read predicted labels from the predicted_labels_file
    with open(predicted_labels_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            left, right = list(map(float, line.split(",")))

            if not left and not right:
                predicted_labels.append(0)
                continue

            if left > threshold:
                predicted_labels.append(-1)
            else:
                predicted_labels.append(1)

    return np.array(true_labels), np.array(predicted_labels)


if __name__ == "__main__":
    true, pred = calculate_mse("./BCICIV_eval_ds1_1000Hz_true_y.txt", "./output_subject_a_1000Hz.csv", 0.571567541421737)

    mse = []
    # count = 0
    for t, p in zip(true, pred[:true.shape[0]]):
        # if count == 1759140:
        #     break
        if t == float("inf"):
            continue
        mse.append((t - p) ** 2)
        # count += 1

    print(np.mean(mse))
