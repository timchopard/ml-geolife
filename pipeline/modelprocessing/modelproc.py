import os
import pickle
import datetime

def save_model(model, model_type: str, note: str = None, pred_cols = None):
    build_dirs(model_type)
    now_string = datetime.datetime.now().strftime("%H-%M_%d-%m")
    note = '' if note is None else note
    full_path = os.path.join("models", model_type, f"{now_string}{note}.pkl")
    pickle.dump(model, open(full_path, "wb"))
    print(f":: Saved model to {full_path}")
    if pred_cols is not None:
        pickle.dump(pred_cols, open(full_path[:-4] + '_cols.pkl', "wb"))

def build_dirs(model_type):
    for dir in ["models/", f"models/{model_type}"]:
        if not os.path.exists(dir):
            print(f":: Creating directory {dir}")
            os.mkdir(dir)

def load_model(model_type: str, model_name: str):
    if len(model_name) < 4 or model_name[-4:] != '.pkl':
        model_name += '.pkl'
    path = os.path.join("models", model_type, model_name)
    return pickle.load(open(path, 'rb'))