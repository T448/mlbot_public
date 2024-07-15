import datetime
from typing import Dict
import mlflow


def get_tscv_index_dict(run_id: str) -> Dict[int, Dict[str, datetime.datetime]]:
    tscv_idx_dict: Dict[int, Dict[str, datetime.datetime]] = {}

    client = mlflow.tracking.MlflowClient()
    for idx_name in ["train_start_index", "train_end_index", "test_start_index", "test_end_index"]:
        hists = client.get_metric_history(run_id, idx_name)
        for hist in hists:
            if hist.step in tscv_idx_dict:
                tscv_idx_dict[hist.step][idx_name] = datetime.datetime.fromtimestamp(hist.value)
            else:
                tscv_idx_dict[hist.step] = {idx_name: datetime.datetime.fromtimestamp(hist.value)}

    return tscv_idx_dict


def get_eval_index(run_id: str) -> Dict[str, datetime.datetime]:
    client = mlflow.tracking.MlflowClient()
    eval_start_index = client.get_metric_history(run_id, "eval_start_index")
    eval_end_index = client.get_metric_history(run_id, "eval_end_index")

    return {
        "eval_start_index": datetime.datetime.fromtimestamp(eval_start_index[0].value),
        "eval_end_index": datetime.datetime.fromtimestamp(eval_end_index[0].value),
    }
