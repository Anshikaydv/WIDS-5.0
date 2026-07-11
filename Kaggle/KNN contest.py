import numpy as np
import pandas as pd

train = pd.read_csv("/kaggle/input/wi-ds-dl-competition-1/train.csv")
test  = pd.read_csv("/kaggle/input/wi-ds-dl-competition-1/test.csv")

X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_test = test.values

X_train = X_train / 255.0
X_test = X_test / 255.0

batch_size = 500
predictions = []

for start in range(0, X_test.shape[0], batch_size):
    end = start + batch_size
    X_batch = X_test[start:end]

    dists = (
        np.sum(X_batch**2, axis=1).reshape(-1, 1)
        + np.sum(X_train**2, axis=1)
        - 2 * np.dot(X_batch, X_train.T)
    )

    nearest = np.argmin(dists, axis=1)
    predictions.extend(y_train[nearest])

    print(f"Processed {len(predictions)} samples")

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv("submission.csv", index=False)

print(X_test.shape)
submission.shape
print(len(predictions))
print("Using predictions length:", len(predictions))
