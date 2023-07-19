from classifiers import classifiers, gzip_classify
from regressors import regressors, gzip_regress


class Models:
    def __init__(self):
        self.classifiers = classifiers
        self.regressors = regressors

    def get_classifier(self, name):
        return self.classifiers[name]

    def get_regressor(self, name):
        return self.regressors[name]

    def task(self, task: str):
        if task == "classification":
            return self.classifiers
        elif task == "regression":
            return self.regressors
        else:
            raise ValueError(f"Unknown task {task}.")