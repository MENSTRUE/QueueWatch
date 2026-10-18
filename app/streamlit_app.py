from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from src.inference import QueueWatchInference
from src.utils import fold_predictions_dataframe

MODEL_DIR=PROJECT_ROOT/"models"
EXPECTED_MODELS=[MODEL_DIR/f"queuewatch_fold{fold}_best.pt" for fold in range(5)]

OOF_MAE=41.8053436
OOF_RMSE=65.5135
NAIVE_MAE=74.260
RELATIVE_MAE_REDUCTION=43.7

st.set_page_config(page_title="QueueWatch",page_icon="👥",layout="wide")
st.title("👥 QueueWatch")
st.caption("Crowd-count regression using a custom-trained 5-fold MobileNetV3-Small ensemble.")

st.info(
    "QueueWatch estimates the number of visible people in an image. "
    "It is a general crowd-counting model, not a queue-specific production system."
)

missing=[p.name for p in EXPECTED_MODELS if not p.exists()]
if missing:
    st.error("Missing trained fold models in `models/`: "+", ".join(missing))
    st.stop()

with st.sidebar:
    st.header("Model")
    st.write("**Backbone:** MobileNetV3-Small")
    st.write("**Target:** log1p(person count)")
    st.write("**Input:** 320 × 320")
    st.write("**Validation:** 5-fold CV")
    st.write("**Ensemble:** median of 5 folds")
    st.divider()
    st.subheader("Final OOF benchmark")
    st.metric("OOF MAE",f"{OOF_MAE:.3f}")
    st.metric("OOF RMSE",f"{OOF_RMSE:.3f}")
    st.caption(
        f"Naive fold-median MAE: {NAIVE_MAE:.3f}. "
        f"Relative MAE reduction: ≈{RELATIVE_MAE_REDUCTION:.1f}%."
    )
    st.warning("MAE/RMSE are counting errors, not accuracy percentages.")

@st.cache_resource(show_spinner="Loading QueueWatch fold models...")
def load_inference():
    device="cuda" if torch.cuda.is_available() else "cpu"
    return QueueWatchInference(model_dir=MODEL_DIR,device=device,image_size=320)

uploaded=st.file_uploader(
    "Upload a crowd image",
    type=["jpg","jpeg","png","webp","bmp"],
)

if uploaded is None:
    st.info("Upload one image to estimate the visible crowd count.")
    st.stop()

try:
    image=Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Unable to open image: {exc}")
    st.stop()

left,right=st.columns([1.2,1.0])

with left:
    st.subheader("Input image")
    st.image(image,use_container_width=True)

with st.spinner("Running 5-fold ensemble inference..."):
    inference=load_inference()
    result=inference.predict(image)

with right:
    st.subheader("Estimated crowd")
    st.metric("Median ensemble estimate",f"{result.rounded_count} people")
    st.metric("Raw median",f"{result.median:.2f}")
    st.metric("Fold prediction range",f"{result.minimum:.1f} – {result.maximum:.1f}")
    st.metric("Fold spread",f"{result.spread:.2f}")
    st.caption(
        "Fold spread measures disagreement among CV models. "
        "It is not a confidence percentage."
    )

st.divider()
st.subheader("Fold predictions")

fold_df=fold_predictions_dataframe(result)

st.dataframe(
    fold_df,
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(
    fold_df[["fold","estimated_count"]].set_index("fold")
)

st.subheader("Ensemble diagnostics")

diagnostics=pd.DataFrame([
    {"metric":"Median","value":result.median},
    {"metric":"Mean","value":result.mean},
    {"metric":"Minimum fold","value":result.minimum},
    {"metric":"Maximum fold","value":result.maximum},
    {"metric":"Fold spread","value":result.spread},
    {"metric":"Fold standard deviation","value":result.std},
])

diagnostics["value"]=diagnostics["value"].round(3)

st.dataframe(
    diagnostics,
    use_container_width=True,
    hide_index=True,
)

st.subheader("How to interpret this result")

st.markdown(
    f"""
- The headline estimate is the **median** of the five fold predictions.
- The model predicts **person count**, not bounding boxes or a density map.
- A larger fold spread means the ensemble members disagree more strongly.
- Final benchmark: **OOF MAE {OOF_MAE:.3f}** and **OOF RMSE {OOF_RMSE:.3f}**.
- These are regression errors and should not be described as accuracy.
"""
)

st.download_button(
    "Download fold predictions CSV",
    data=fold_df.to_csv(index=False).encode("utf-8"),
    file_name="queuewatch_fold_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)
