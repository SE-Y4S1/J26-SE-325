import numpy as np
import shap

from .fraud_model import model


FEATURE_NAMES = [
    "new_device",
    "unusual_location",
    "amount_thousands"
]


# Create SHAP TreeExplainer
explainer = shap.TreeExplainer(model)


def explain_transaction(
    new_device: int,
    unusual_location: int,
    amount_lkr: float
):
    """
    Generate SHAP explanations for the fraud class.
    """

    amount_thousands = amount_lkr / 1000

    features = np.array([
        [
            new_device,
            unusual_location,
            amount_thousands
        ]
    ])

    # Generate SHAP values
    shap_values = explainer.shap_values(features)

    # Current SHAP output for our Random Forest:
    #
    # shape = (samples, features, classes)
    #
    # We need class 1 = fraud.
    values = shap_values[0, :, 1]

    explanation = []

    for feature_name, value in zip(
        FEATURE_NAMES,
        values
    ):

        explanation.append(
            {
                "feature": feature_name,
                "shap_value": float(value),
                "impact": (
                    "increases fraud risk"
                    if value > 0
                    else "decreases fraud risk"
                )
            }
        )

    # Strongest contributors first
    explanation.sort(
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    # Model prediction
    probability = model.predict_proba(features)[0][1]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),

        "features": {
            "new_device": new_device,
            "unusual_location": unusual_location,
            "amount_lkr": amount_lkr
        },

        "shap_explanation": explanation
    }