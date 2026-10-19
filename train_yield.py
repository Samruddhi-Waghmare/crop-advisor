import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

df = pd.read_csv("data/crop_production.csv")

df = df.dropna(subset=["Production"])

df = df.dropna(subset=["Production"])
for col in ["State_Name","Season","Crop"]:
    df[col] = df[col].str.strip()
df = df[df["Area"] > 0]   
df = df[df["Crop"] != "Coconut"]
df["Yield"] = df["Production"]/df["Area"]
df = df[df["Yield"] < df["Yield"].quantile(0.99)]
print("Rows after cleaning:", len(df))

features = ["State_Name", "Season", "Crop", "Crop_Year", "Area"]
cat_cols = ["State_Name","Season","Crop"]
num_cols = ["Crop_Year", "Area"]

X = df[features]
y =df["Yield"]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

def make_model(regressor):
    # Yield is heavily skewed, so train on log(yield) and convert back afterwards
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("model", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )

sample = df.sample(min(10000, len(df)), random_state=42)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    sample[features], sample["Yield"], test_size=0.2, random_state=42
)

for name, reg in [
    ("KNN", KNeighborsRegressor(n_neighbors = 7)),
    ("XGBoost", XGBRegressor(n_estimators=400, learning_rate=0.1,
    max_depth=8, random_state=42)),
]:
    m = make_model(reg)
    m.fit(Xs_train, ys_train)
    pred = m.predict(Xs_test)
    rmse = np.sqrt(mean_squared_error(ys_test, pred))
    print(f"{name}: R2 = {r2_score(ys_test, pred):.3f} | RMSE = {rmse:.3f}")

# ---------- Train final XGBoost on ALL data and save ----------
final = make_model(XGBRegressor(n_estimators=400, learning_rate=0.1,
                                max_depth=8, random_state=42))
final.fit(X, y)
joblib.dump(final, "models/yield_xgb.pkl")

# Lists the app needs for its dropdowns
meta = {
    "states": sorted(df["State_Name"].unique().tolist()),
    "seasons": sorted(df["Season"].unique().tolist()),
    "crops": sorted(df["Crop"].unique().tolist()),
    "latest_year": int(df["Crop_Year"].max()),
}
joblib.dump(meta, "models/meta.pkl")
print("Saved!")


