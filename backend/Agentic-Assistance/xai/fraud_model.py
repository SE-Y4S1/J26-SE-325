import numpy as np

from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# MOCK TRAINING DATA
# --------------------------------------------------

# Features:
#
# 0 = normal device
# 1 = new device
#
# 0 = normal location
# 1 = unusual location
#
# amount is represented in thousands of LKR
#
# Example:
# [new_device, unusual_location, amount_in_thousands]

X_train = np.array([
    [0, 0, 20],
    [0, 0, 50],
    [0, 0, 80],
    [0, 1, 40],
    [1, 0, 30],
    [1, 1, 40],
    [1, 1, 100],
    [0, 0, 120],
    [0, 1, 150],
    [1, 0, 200],
    [1, 1, 250],
    [1, 1, 300],
])

# 0 = legitimate
# 1 = suspicious/fraud

y_train = np.array([
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
])


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_fraud(
    new_device: int,
    unusual_location: int,
    amount_lkr: float
):
    """
    Predict fraud probability for a transaction.

    Parameters:
        new_device:
            0 = normal device
            1 = new device

        unusual_location:
            0 = normal location
            1 = unusual location

        amount_lkr:
            Transaction amount in LKR.
    """

    amount_thousands = amount_lkr / 1000

    features = np.array([
        [
            new_device,
            unusual_location,
            amount_thousands
        ]
    ])

    probability = model.predict_proba(features)[0][1]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "features": {
            "new_device": new_device,
            "unusual_location": unusual_location,
            "amount_lkr": amount_lkr
        }
    }