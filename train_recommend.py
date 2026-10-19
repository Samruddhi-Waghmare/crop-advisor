import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/Crop_recommendation.csv")
X = df.drop("label", axis = 1)

# we convert crop names to numbers so use label encoder 
le = LabelEncoder()
y = le.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify=y
)

# KNN -- uses distance so features can be scaled 
# without scaling, rainfall ranges from (0-300) would dominate pH(3-9)

knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = GridSearchCV(
    knn_pipe,
    {"kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 11]},
    cv=5,
)
grid.fit(X_train, y_train)
knn = grid.best_estimator_
print("Best K:", grid.best_params_)
knn_pred = knn.predict(X_test)
print("KNN accuracy:", accuracy_score(y_test, knn_pred))


# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="mlogloss",
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGBoost accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred, target_names=le.classes_))

joblib.dump(xgb, "models/recommend_xgb.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("Saved!")