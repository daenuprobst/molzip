from benchmark import benchmark
from configs import dataset_configs
from models import Models, gzip_classify, gzip_regress


def main(model=None, preprocess_task="smiles", tasks="all"):
    models = Models()
    regress = None
    classify = None

    #  create a list of models, classifiers and regressors
    if model is None:
        models = ["gzip"]
        classify = gzip_classify
        regress = gzip_regress
    elif isinstance(model, str):
        classify = [models.classifiers[model]]
        regress = [models.regressors[model]]
        models = [model]
    elif isinstance(model, list):
        #  assume a list of strings
        classify = [models.classifiers[m] for m in model]
        regress = [models.regressors[m] for m in model]
        models = model

    if tasks == "all":
        print("Performing regression and classification benchmarks")
    elif tasks == "regression":
        print("Performing regression benchmarks")
        classify = [None for _ in classify]
    elif tasks == "classification":
        print("Performing classification benchmarks")
        regress = [None for _ in regress]

    # loop through the models and run the benchmark
    for model, classifier, regressor, in zip(models, classify, regress):
        benchmark(
            dataset_configs,
            model=model,
            classify=classifier,
            regress=regressor,
            preprocess_task=preprocess_task,
        )


if __name__ == "__main__":
    """ Run the benchmarks
    """
    from argparse import ArgumentParser

    # command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--tasks", type=str, default="all")
    arg_parser.add_argument("--model", type=str, default="gzip")
    args = arg_parser.parse_args()
    main(
        model=args.model,
        tasks=args.tasks
    )
