import pandas as pd

def fold_predictions_dataframe(result):
    return pd.DataFrame([
        {
            "fold":f"Fold {item.fold+1}",
            "estimated_count":round(item.count_prediction,2),
            "log1p_prediction":round(item.log_prediction,5),
        }
        for item in result.fold_predictions
    ])
