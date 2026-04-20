from config import pd

def load_data():
    path = "used_cars.csv"
    df = pd.read_csv(path)
    return df