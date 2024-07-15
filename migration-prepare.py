import optuna
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO


def objective(trial):
    u = trial.suggest_float("u", -100, 100)
    lu = trial.suggest_float("lu", 1, 100, log=True)
    du = trial.suggest_float("du", -100, 100, step=2)

    i = trial.suggest_int("i", -100, 100)
    li = trial.suggest_int("li", 1, 100, log=True)
    si = trial.suggest_int("si", -100, 100, step=2)

    c = trial.suggest_categorical("c", [-1, 0, 1])
    for epoch in range(10):
        trial.report(epoch * 2, epoch)

    trial.set_user_attr("user_key", "user_value")
    trial.set_system_attr("system_key", "system_value")

    return sum([u, lu, du, i, li, si, c])


def plot_result(x, filename=None):
    plt.clf()
    plt.figure()
    plt.plot(x, "-o")

    img = BytesIO()
    plt.savefig(img)

    return Image.open(img)


if __name__ == "__main__":
    print(f"optuna version: {optuna.__version__}")

    x = np.random.normal(0, 1, 1024)

    with mlflow.start_run(run_name="migration"):
        study = optuna.create_study(study_name="migration")

        study.set_user_attr("study_user_key", "study_user_value")
        study.set_system_attr("study_system_key", "study_system_value")

        study.optimize(objective, n_trials=20)

        mlflow.log_artifact(__file__, "src")
        mlflow.log_metric("param1", 1)

        print(f"Best value: {study.best_value} (params: {study.best_params})\n")

        mlflow.log_image(plot_result(x), "result.png")
