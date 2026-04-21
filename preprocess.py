import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def set_seed(seed=42):
    np.random.seed(seed)


def process_data(df):
    set_seed()
    X = df.drop(["price"], axis=1)
    y = np.log(df["price"])

    num_feats = ["milage", "car_age"]
    cat_feats = ["brand", "model", "fuel_type", "transmission"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test, preprocessor