import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("coffee_sales_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.columns)


# --------------------------------------------------
# 2. Select Target
# --------------------------------------------------

# Target variable (What we want to predict)
target = "TotalBillAmount"

# Remove rows where target is missing
df = df.dropna(subset=[target])


# --------------------------------------------------
# 3. Select Features
# --------------------------------------------------

features = [
    "QuantityOrdered",
    "CoffeeType",
    "CoffeeVariety",
    "Size",
    "CustomerType",
    "PaymentMethod"
]

X = df[features]
y = df[target]


# --------------------------------------------------
# 4. Numerical & Categorical Columns
# --------------------------------------------------

numeric_features = [
    "QuantityOrdered"
]

categorical_features = [
    "CoffeeType",
    "CoffeeVariety",
    "Size",
    "CustomerType",
    "PaymentMethod"
]


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            "passthrough",
            numeric_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 6. Model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 7. Create Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 9. Train Model
# --------------------------------------------------

pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 10. Prediction
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 11. Model Evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\nModel Performance")
print("------------------")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# --------------------------------------------------
# 12. Save Model
# --------------------------------------------------

with open(
    "coffee_sales_model.pkl",
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )


print("\nModel saved successfully!")

print("Model File: coffee_sales_model.pkl")
