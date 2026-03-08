import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess_gpu_data


def train():
    df = preprocess_gpu_data()

    numeric_features_final = df.select_dtypes(
        include=["float64"]).drop(columns=["Avg_FPS"]).columns.tolist()

    categorical_features = ["Resolution",
                            "Setting", "memory_type", "architecture", "Game_Name"]

    X = df[numeric_features_final + categorical_features]
    y = df["Avg_FPS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features_final),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # XGBoost pipeline
    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor())
    ])

    # Fit and evaluate
    xgb_pipeline.fit(X_train, y_train)

    joblib.dump(xgb_pipeline, "models/xgb_gpu_fps_model.joblib")
