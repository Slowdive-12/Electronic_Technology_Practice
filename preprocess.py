from config import train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer

def process_data(df):
    df = df[(df["price"] > 0) & (df["milage"] > 0)]
    df["car_age"] = 2026 - df["model_year"]

    X = df.drop(["price"], axis=1)
    y = df["price"]

    cat_feats = ["brand", "model", "fuel_type", "transmission"]
    num_feats = ["milage", "car_age"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test