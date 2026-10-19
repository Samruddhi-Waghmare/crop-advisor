import numpy as np
import pandas as pd 
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/Crop_recommendation.csv")
X = df.drop("label", axis=1)
model = joblib.load("models/recommend_xgb.pkl")
le = joblib.load("models/label_encoder.pkl")
y= le.transform(df["label"])

# Same split as training (same random_state), so we explain on unseen data
_, X_test, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Calculate SHAP values ----------
explainer = shap.TreeExplainer(model)
sv = np.array(explainer.shap_values(X_test))
print("SHAP array shape:", sv.shape)


# Newer SHAP returns (samples, features, classes); older returns (classes, samples, features)
crop = "rice"
idx = list(le.classes_).index(crop)
if sv.shape[0] == len(le.classes_):
    sv_crop = sv[idx]          # old layout
else:
    sv_crop = sv[:, :, idx]    # new layout

# ---------- Plot 1: why does the model recommend rice? ----------
shap.summary_plot(sv_crop, X_test, show=False)
plt.title(f"What drives the model to recommend {crop.upper()}?")
plt.tight_layout()
plt.savefig("shap_rice.png", dpi=150)
plt.show()

# ---------- Plot 2: overall feature importance ----------
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
imp.plot.barh(color="green")
plt.title("Overall feature importance (XGBoost)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
