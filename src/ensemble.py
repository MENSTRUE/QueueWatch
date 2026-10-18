from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .model import build_queuewatch_model

@dataclass(slots=True)
class FoldPrediction:
    fold:int
    log_prediction:float
    count_prediction:float

@dataclass(slots=True)
class EnsemblePrediction:
    count:float
    rounded_count:int
    fold_predictions:list[FoldPrediction]
    minimum:float
    maximum:float
    spread:float
    std:float
    mean:float
    median:float

class QueueWatchEnsemble:
    def __init__(self,model_dir,device=None,checkpoint_pattern="queuewatch_fold{fold}_best.pt",num_folds=5):
        self.model_dir=Path(model_dir)
        self.checkpoint_pattern=checkpoint_pattern
        self.num_folds=int(num_folds)
        self.device=torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models=[]
        self._load_models()

    def _path(self,fold):
        return self.model_dir/self.checkpoint_pattern.format(fold=fold)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint,dict):
            for key in ("model_state_dict","state_dict","model"):
                value=checkpoint.get(key)
                if isinstance(value,dict):
                    return value
        return checkpoint

    @staticmethod
    def _strip_module_prefix(state_dict):
        if not isinstance(state_dict,dict):
            raise TypeError("Checkpoint does not contain a state_dict.")
        if any(k.startswith("module.") for k in state_dict):
            return {k.removeprefix("module."):v for k,v in state_dict.items()}
        return state_dict

    def _load_models(self):
        missing=[self._path(f) for f in range(self.num_folds) if not self._path(f).exists()]
        if missing:
            raise FileNotFoundError("Missing fold checkpoint(s):\n"+"\n".join(f"- {p}" for p in missing))

        for fold in range(self.num_folds):
            model=build_queuewatch_model(pretrained=False)
            checkpoint=torch.load(self._path(fold),map_location=self.device,weights_only=False)
            state=self._strip_module_prefix(self._extract_state_dict(checkpoint))
            model.load_state_dict(state,strict=True)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    @torch.inference_mode()
    def predict(self,image_tensor):
        if image_tensor.ndim==3:
            image_tensor=image_tensor.unsqueeze(0)
        if image_tensor.ndim!=4 or image_tensor.shape[0]!=1:
            raise ValueError("Expected one image tensor [C,H,W] or [1,C,H,W].")

        x=image_tensor.to(self.device,non_blocking=True)
        folds=[]
        counts=[]

        for fold,model in enumerate(self.models):
            raw=model(x).reshape(-1)[0]
            log_pred=float(raw.detach().cpu().item())
            count=max(0.0,float(np.expm1(log_pred)))
            folds.append(FoldPrediction(fold,log_pred,count))
            counts.append(count)

        arr=np.asarray(counts,dtype=np.float64)
        median=float(np.median(arr))
        return EnsemblePrediction(
            count=median,
            rounded_count=max(0,int(round(median))),
            fold_predictions=folds,
            minimum=float(arr.min()),
            maximum=float(arr.max()),
            spread=float(arr.max()-arr.min()),
            std=float(arr.std()),
            mean=float(arr.mean()),
            median=median,
        )
