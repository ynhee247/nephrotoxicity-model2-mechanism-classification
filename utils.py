import joblib

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str): # nạp lại model (k cần train lại)
    return joblib.load(path)