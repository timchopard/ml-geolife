""" python ensemblexgb.py [--train --load <model-name> <count-name> --run]
"""
import sys

from xgboost import XGBRegressor

from pipeline.preprocess import PCAApplier
from pipeline.modelprocessing import save_model, load_model
from pipeline.postprocessing import Postprocessor
from helper.top_n_regression import get_top_n

SEED = 31

class XGBPipeline():
    """ Creates and trains, or loads in XGBoost models and runs the training 
    data through them
    """

    model = None
    count_model = None
    column_labels = None
    model_type = "xgbregressor"

    def train_models(self, device="cuda"):
        """ Train the XGBoost models. Loads in the training data, creates the
        main and count models and then fits them and saves them.

        args:
            device      :   "cuda" to use gpu, else "cpu" to use the cpu
        """
        train = PCAApplier('data/processed/train_full.pkl', method='full').data
        xx_cols = [col for col in train.columns if "species" not in col]
        yy_cols = [col for col in train.columns if "species" in col]
        self.model = XGBRegressor(device=device)
        self.count_model = XGBRegressor(device=device)
        print(":: Fitting main model")
        self.model.fit(train[xx_cols], train[yy_cols])
        save_model(self.model, "xgbregressor", "-full-main", yy_cols)
        print(":: Fitting species count model")
        self.count_model.fit(
            train[xx_cols],
            train[yy_cols].apply(lambda x: x.sum(), axis=1)
        )
        save_model(self.count_model, self.model_type, "-full-counts")
        self.column_labels = yy_cols

    def load_models(self, model_name, count_name):
        """ Loads in the pretrained models and stores them in member variables
        note: the filenames work with or without the .pkl extension, the 
        directory path is not needed
        
        args:
            model_name  :   The file name of the main model
            count_name  :   The file name of the count model
        """
        print(f":: Loading {model_name} as main model")
        self.model = load_model(self.model_type, model_name)
        print(f":: Loading {model_name} as species count model")
        self.count_model = load_model(self.model_type, count_name)
        self.column_labels = load_model(self.model_type, f"{model_name}_cols")


    def run(self, additional_count: int = None):
        """ Runs the models on the test data. Processes the data into uploadable
        format and saves it in the submissions directory

        args:
            additional_count    :   An amount to add to the estimated count
                                    of species selected per survey
        """
        additional_count = 5 if additional_count is None else additional_count
        if self.model is None or self.count_model is None:
            raise Exception("[ERROR] No model currently available")

        test = PCAApplier(method='full').data

        liklihoods = self.model.predict(test)
        occurences = self.count_model.predict(test).round().astype(int)          \
                   + additional_count

        Postprocessor(
            predictions=get_top_n(liklihoods, occurences),
            pred_indices=test.index,
            pred_cols=self.column_labels,
            save=True,
            uploadable=True
        )


if __name__ == "__main__":
    args = sys.argv
    accepted_args = bool(
        [aa for aa in args if aa in["--train", "--load", "--run"]]
    )
    if len(args) == 1 or "-h" in args or not accepted_args:
        print("This program runs the XGBoost Classifier to generate probable " \
            + "species for each survey as well as the number of species likely"\
            + " to be present in the survey.\n\t- run with \'--train'\tto"     \
            + " train and save the models\n\t- run with \'--load "             \
            + "<model-name> <count-model-name>\'\tto load in and process an "  \
            + "existing model")
        sys.exit(0)

    pipeline = XGBPipeline()

    if "--train" in args:
        pipeline.train_models()

    if "--load" in args:
        load_index = args.index("--load")
        if len(args) < load_index + 3:
            print("Please use the \'--load\' flag in the following format:\n"  \
                + "\tpython ensemblexgb.py --load modelName countModelName")
        pipeline.load_models(args[load_index + 1], args[load_index + 2])

    if "--run" in args:
        if "--train" not in args and "--load" not in args:
            print("No model present, run with either \'--train\' or \'--load\'"\
                + " as well as the \'--run\' flag")
            sys.exit(0)

        pos = args.index("--run") 
        count = None
        if len(args) > pos + 1 and args[pos + 1].isnumeric():
            count = int(args[pos + 1])

        pipeline.run(count)
