import numpy as np
import shap

from .fraud_model import model


features = np.array([
    [1, 1, 250]
])

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(features)

print("SHAP type:")
print(type(shap_values))

print("\nSHAP values:")
print(shap_values)

print("\nSHAP shape:")
print(np.array(shap_values).shape)

print("\nExpected value:")
print(explainer.expected_value)

print("\nModel probabilities:")
print(model.predict_proba(features))