import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time


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


def simulate_labels(true_labels, predicted_labels):
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(true_labels))
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Time')
    ax.set_ylabel('Label')
    ax.set_title('True Labels vs Predicted Labels')

    true_line, = ax.plot([], [], label='True Labels')
    predicted_line, = ax.plot([], [], label='Predicted Labels')
    ax.legend()

    x = np.arange(len(true_labels))

    def update(frame):
        x = np.arange(frame + 1)  # Update x-axis values based on frame
        ax.set_xlim(0, frame + 1)  # Update x-axis limits

        true_line.set_data(x, true_labels[:frame + 1])
        predicted_line.set_data(x, predicted_labels[:frame + 1])
        return true_line, predicted_line

    ani = FuncAnimation(fig, update, frames=len(true_labels), interval=50, blit=True)
    ax.plot(true_labels, 'r')
    ax.plot(predicted_labels, 'b')

    plt.show()


if __name__ == "__main__":

    maxMSEPairsX, maxMSEPairsY = [], []
    minMSE = float("inf")
    bestThreshold = None

    for thresholdVal in np.arange(0, 1, 0.05):
        true, pred = calculate_mse("./BCICIV_eval_ds1f_1000Hz_true_y.txt", "./output_subject_f_1000Hz.csv",
                                   thresholdVal)
        mse = []
        # count = 0

        for t, p in zip(true, pred[:true.shape[0]]):
            # if count == 1759140:
            #     break
            if t == float("inf"):
                continue
            mse.append((t - p) ** 2)
            # count += 1

        currMSE = np.mean(mse)
        if currMSE < minMSE:
            minMSE = currMSE
            bestThreshold = thresholdVal

        maxMSEPairsX.append(thresholdVal)
        maxMSEPairsY.append(currMSE)

    plt.plot(maxMSEPairsX, maxMSEPairsY)
    plt.axvline(x=bestThreshold, color='red', ls='--', lw=1.0, label=f"mse:{minMSE}, threshold:{bestThreshold}")
    plt.legend()
    plt.show()

    # Example usage
    # true_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])  # Example true labels
    # predicted_labels = np.array([0, 0, 1, 0, 1, 0, 0, 1])  # Example predicted labels
    #
    # simulate_labels(true, pred)
    # print(np.mean([1.473, 0.907, 0.491, 1.502]))
