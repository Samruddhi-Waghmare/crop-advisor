import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rec = pd.read_csv("data/Crop_recommendation.csv")
print("Shape (rows,columns):", rec.shape)
print(rec.head())
print("\nMissing values:\n", rec.isnull().sum())
print("\nRows per crop:\n",rec["label"].value_counts())

plt.figure(figsize=(8,6))
sns.heatmap(rec.drop("label",axis=1).corr(), annot=True, cmap="Greens")
plt.title("Feature correlation")
plt.show()

prod = pd.read_csv("data/crop_production.csv")
print("\nShape:", prod.shape)
print("\nMissing values:\n", prod.isnull().sum())
print("\nSeasons:", prod["Season"].unique())
print("\nCrops:", prod["Crop"].unique())   # note these names, we need them later

prod["Yield"] = prod["Production"] / prod["Area"]
print("\nYield summary:\n", prod["Yield"].describe())