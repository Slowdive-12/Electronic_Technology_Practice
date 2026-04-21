def get_loader(X_train, X_test, y_train, y_test, batch_size=32):
    return (X_train, y_train), (X_test, y_test)