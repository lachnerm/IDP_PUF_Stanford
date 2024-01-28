import os

import numpy as np
from argparse import Namespace

import optuna
from optuna.integration import TensorBoardCallback

from main import run_model


def objective(trial):
    bs = trial.suggest_categorical("bs", [32, 64, 128, 256])
    ns = trial.suggest_int("hidden_size", 50, 5000)
    n_layer = trial.suggest_int("n_layer", 1, 5)
    lr = trial.suggest_categorical("lr", [1e-1, 1e-2, 1e-3, 1e-4])

    hparams = {
        "bs": bs,
        "lr": lr,
        "ns": ns,
        "n_layer": n_layer
    }

    args = Namespace(**{
        'store_results': False,
        'epochs': 100,
        'cbits': 64,
        'hparams': hparams
    })

    np.random.seed(0)
    length = 4096
    length = 100000
    ids = list(range(length))
    np.random.shuffle(ids)
    train_ids = ids[:int(length * 0.8)]
    val_ids = ids[len(train_ids):int(length * 0.9)]
    test_ids = ids[len(train_ids) + len(val_ids):]

    data_dir = f'data/SwitchableStarPUF/12bit_enumerate_0.csv'
    data_dir = f'data/SwitchableStarPUF/64bit_rand100k.csv'
    model = run_model(args, data_dir, (train_ids, val_ids, test_ids))

    return np.max(model.val_accs)


if __name__ == "__main__":
    if not os.path.exists('optim.db'):
        study = optuna.create_study(
            study_name="optim",
            storage="sqlite:///optim.db",
            direction="maximize"
        )
    study = optuna.load_study(
        study_name="optim",
        storage="sqlite:///optim.db"
    )

    tensorboard_callback = TensorBoardCallback("logs/", metric_name="value")

    study.optimize(objective, n_trials=100, callbacks=[tensorboard_callback])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    trials = study.best_trials
    print(f"Best trials: ({len(trials)})")
    for trial in trials:
        print(trial.number)
        print(f"  Val loss:: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
