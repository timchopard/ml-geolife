import os
import pickle
import datetime

def save_model(model, model_type):
    build_dirs(model_type)
    now_string = datetime.datetime.now().strftime("%H-%M_%d-%m")
    full_path = os.path.join("models", model_type, f"{now_string}.pkl")
    pickle.dump(model, open(full_path, "wb"))
    print(f":: Saved model to {full_path}")

def build_dirs(model_type):
    for dir in ["models/", f"models/{model_type}"]:
        if not os.path.exists(dir):
            print(f":: Creating directory {dir}")
            os.mkdir(dir)