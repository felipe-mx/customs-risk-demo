# app.py
# Streamlit demo for Customs Illicit-Risk Scoring
# - Upload a model artifact (model_pipeline.joblib) OR train a quick model in-app
# - Upload a CSV/XLSX with columns (broker, hts, description, unit_price, customs_value,
#   commercial_value, quantity, country_origin, date, illicit_label[optional for scoring])
# - Get risk scores, top incidents, and optional explanations via SHAP

import io, json, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
import joblib

from pathlib import Path
DEFAULT_MODEL_PATH = Path("artifact/model_pipeline.joblib")


st.set_page_config(page_title="Customs Risk Prototype", layout="wide")

# ------------------------
# Feature Engineering (same as training)
# ------------------------
REQUIRED_FOR_SCORING = [
    "broker","hts","unit_price","customs_value","commercial_value",
    "quantity","country_origin","date"
]
CAT_COLS = ["broker","hts","country_origin"]
NUM_COLS = [
    "unit_price","customs_value","commercial_value","quantity",
    "price_gap_ratio","unit_price_dev_hs_origin",
    "qty_is_one","qty_le_2",
    "log_unit_price","log_customs_value","log_commercial_value","log_quantity",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    eps = 1e-9
    df["price_gap_ratio"] = (df["customs_value"] - df["commercial_value"]) / (
        df["commercial_value"].replace(0, np.nan) + eps
    )

    grp = df.groupby(["hts","country_origin"])["unit_price"]
    median = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median())
    mad = mad.replace(0, np.nan).fillna(1.0)
    df["unit_price_dev_hs_origin"] = (df["unit_price"] - median) / mad

    df["qty_is_one"] = (df["quantity"] == 1).astype(int)
    df["qty_le_2"]  = (df["quantity"] <= 2).astype(int)

    for c in ["unit_price","customs_value","commercial_value","quantity"]:
        df[f"log_{c}"] = np.log(df[c].astype(float) + 1.0)

    X = pd.concat(
        [df[CAT_COLS].astype(str).fillna("NA"),
         df[NUM_COLS].astype(float).fillna(0.0)],
        axis=1
    )
    return X

def make_preprocessor(n_cat: int, n_total: int):
    # sklearn v1.4+ uses sparse_output; older uses sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("cat", ohe, list(range(n_cat))),
            ("num", "passthrough", list(range(n_cat, n_total))),
        ]
    )

def train_quick_model(df_all: pd.DataFrame):
    # expects columns incl. illicit_label
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["illicit_label"] = df["illicit_label"].astype(int)
    df = df.dropna(subset=["date","illicit_label"])

    # simple time split (last 20% validation)
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df)*0.8) if len(df) > 10 else max(1, len(df)-1)
    train_df, valid_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    X_train = build_features(train_df)
    X_valid = build_features(valid_df)
    y_train = train_df["illicit_label"].values
    y_valid = valid_df["illicit_label"].values

    pre = make_preprocessor(len(CAT_COLS), X_train.shape[1])
    clf = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=400, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)

    valid_proba = pipe.predict_proba(X_valid)[:,1]
    if len(np.unique(y_valid)) > 1:
        auprc = float(average_precision_score(y_valid, valid_proba))
        rocauc = float(roc_auc_score(y_valid, valid_proba))
    else:
        auprc, rocauc = float("nan"), float("nan")

    metrics = {"AUPRC": auprc, "ROC_AUC": rocauc, "n_train": len(train_df), "n_valid": len(valid_df)}
    return pipe, metrics

@st.cache_data(show_spinner=False)
def load_table(upload) -> pd.DataFrame:
    import pandas as pd
    if upload is None:
        return pd.DataFrame()
    name = upload.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(upload)

    if name.endswith(".xlsx"):
        # needs openpyxl in requirements.txt
        return pd.read_excel(upload, engine="openpyxl")

    if name.endswith(".xls"):
        # only if you added xlrd==1.2.0 to requirements
        return pd.read_excel(upload, engine="xlrd")

    st.error("Unsupported file type. Please upload .csv, .xlsx, or .xls")
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model_bytes(model_bytes):
    return joblib.load(io.BytesIO(model_bytes))

st.title("📦 Customs Illicit-Risk — Streamlit Prototype")

# ------------------------
# Sidebar: inputs
# ------------------------
st.sidebar.header("1) Load model or train quickly")
model_file = st.sidebar.file_uploader("Upload model_pipeline.joblib (optional)", type=["joblib","pkl"])
train_if_missing = st.sidebar.checkbox("If no model uploaded, train a quick model from my file (requires illicit_label)", value=True)

st.sidebar.header("2) Upload data to score")
data_file = st.sidebar.file_uploader("Upload CSV/XLSX of customs operations", type=["csv","xlsx","xls"])

top_k = st.sidebar.slider("Show top N incidents", min_value=10, max_value=500, value=50, step=10)
want_importance = st.sidebar.checkbox("Compute global feature importance (slow on big data)", value=False)
want_shap = st.sidebar.checkbox("Add per-shipment explanations for top N (SHAP)", value=False)

# ------------------------
# Load data
# ------------------------
df_raw = load_table(data_file)
if df_raw.empty:
    st.info("Upload a CSV/XLSX with required columns to begin.")
    st.stop()

st.subheader("Raw sample")
st.dataframe(df_raw.head(10), use_container_width=True)

# sanity check on required columns
missing = [c for c in REQUIRED_FOR_SCORING if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns for scoring: {missing}")
    st.stop()

# ------------------------
# Load or train model (prefers a real .joblib if available)
# ------------------------
pipe = None
metrics = {}

if model_file is not None:
    # User uploaded a real artifact
    pipe = load_model_bytes(model_file.read())
    st.success("Loaded model from uploaded .joblib.")
elif DEFAULT_MODEL_PATH.exists():
    # Use repo’s default artifact if present
    import joblib
    pipe = joblib.load(DEFAULT_MODEL_PATH)
    st.success(f"Loaded default model: {DEFAULT_MODEL_PATH}")
elif train_if_missing:
    # Fall back to quick training (requires illicit_label in the data)
    if "illicit_label" not in df_raw.columns:
        st.error("Quick train requires 'illicit_label'. Upload a model or provide labeled data.")
        st.stop()
    with st.spinner("Training quick model..."):
        pipe, metrics = train_quick_model(df_raw)
    st.success("Quick model trained.")
else:
    st.error("Please upload a model (.joblib) or enable quick training.")
    st.stop()

if metrics:
    st.caption("Quick training validation metrics (informational only):")
    st.json(metrics)

# ------------------------
# Scoring with real model (or quick model)
# ------------------------
with st.spinner("Scoring..."):
    df_sc = df_raw.copy()
    X = build_features(df_sc)
    proba = pipe.predict_proba(X)[:, 1]
    df_sc["score_illicit"] = proba
    df_sc["rank"] = (-df_sc["score_illicit"]).rank(method="first").astype(int)
    df_sc = df_sc.sort_values("score_illicit", ascending=False).reset_index(drop=True)

st.subheader(f"Top {top_k} incidents (by score)")
cols_show = ["broker","hts","country_origin","date","unit_price","customs_value","commercial_value","quantity","score_illicit","rank"]
if "illicit_label" in df_sc.columns:
    cols_show = ["illicit_label"] + cols_show
st.dataframe(df_sc[cols_show].head(top_k), use_container_width=True)

# Download scored table
st.download_button(
    "⬇️ Download full scored table (CSV)",
    data=df_sc.to_csv(index=False).encode("utf-8"),
    file_name="scored_incidents.csv",
    mime="text/csv"

# ------------------------
# Global importance (Permutation Importance)
# ------------------------
if want_importance:
    st.subheader("Global feature importance (permutation on avg precision)")
    with st.spinner("Computing permutation importance..."):
        X_imp = build_features(df_sc.head(min(3000, len(df_sc))))  # cap for speed
        y_imp = None
        if "illicit_label" in df_sc.columns and df_sc["illicit_label"].nunique() == 2:
            y_imp = df_sc["illicit_label"].head(len(X_imp)).values
        else:
            st.warning("No binary labels found — importance uses model output stability only.")
            # use pseudo-labels: model scores binarized at median (not ideal, but shows sensitivity)
            pred_imp = pipe.predict_proba(X_imp)[:,1]
            y_imp = (pred_imp >= np.median(pred_imp)).astype(int)

        res = permutation_importance(
            pipe, X_imp, y_imp,
            n_repeats=8, random_state=42, scoring="average_precision"
        )
        pre = pipe.named_steps["pre"]
        try:
            feat_names = pre.get_feature_names_out(X_imp.columns)
        except Exception:
            try:
                ohe = pre.named_transformers_["cat"]
                cat_names = ohe.get_feature_names_out(["broker","hts","country_origin"])
            except Exception:
                cat_names = [f"cat_{i}" for i in range(X_imp.shape[1]-len(NUM_COLS))]
            feat_names = np.concatenate([cat_names, np.array(NUM_COLS)])

        n_model = res.importances_mean.shape[0]
        if len(feat_names) != n_model:
            m = min(len(feat_names), n_model)
            feat_names = feat_names[:m]
            imps = res.importances_mean[:m]
        else:
            imps = res.importances_mean

        imp_df = pd.DataFrame({"feature": feat_names, "importance": imps}).sort_values("importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(7,6))
        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        ax.set_xlabel("Importance"); ax.set_ylabel("Feature"); ax.set_title("Permutation Importance")
        st.pyplot(fig)
        st.dataframe(imp_df, use_container_width=True)

# ------------------------
# Local explanations (SHAP) for top N
# ------------------------
if want_shap:
    st.subheader("Per-shipment explanations (top drivers)")
    with st.spinner("Computing SHAP (top N)…"):
        import shap
        # transformed features
        pre = pipe.named_steps["pre"]
        Z = pre.transform(build_features(df_sc))
        # feature names
        try:
            feat_names = pre.get_feature_names_out(build_features(df_sc).columns)
        except Exception:
            try:
                ohe = pre.named_transformers_["cat"]
                cat_names = ohe.get_feature_names_out(["broker","hts","country_origin"])
            except Exception:
                cat_names = [f"cat_{i}" for i in range(Z.shape[1]-len(NUM_COLS))]
            feat_names = np.concatenate([cat_names, np.array(NUM_COLS)])

        # top N rows indices
        N = min(top_k, len(df_sc))
        top_idx = np.arange(N)

        # background sample for explainer
        bg_size = min(200, Z.shape[0])
        background = shap.sample(Z, bg_size, random_state=42) if Z.shape[0] > bg_size else Z

        f = lambda z: pipe.named_steps["clf"].predict_proba(z)[:,1]
        explainer = shap.Explainer(f, background)
        sv = explainer(Z[top_idx]).values  # (N, n_features)

        def pretty(n: str) -> str:
            n = str(n)
            n = n.replace("x0_", "broker=").replace("x1_", "hts=").replace("x2_", "country=")
            n = n.replace("cat__", "")
            return n

        def drivers(sh, names, k=3):
            order = np.argsort(-np.abs(sh))[:k]
            items = [f"{pretty(names[j])}{'↑' if sh[j]>0 else '↓'}" for j in order]
            return ", ".join(items)

        driver_list = [drivers(sv[i], feat_names, k=3) for i in range(N)]
        df_view = df_sc.iloc[:N].copy()
        df_view["top_drivers"] = driver_list
        st.dataframe(df_view[["broker","hts","country_origin","date","score_illicit","top_drivers"]], use_container_width=True)

        csv = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download top incidents with explanations (CSV)", data=csv, file_name="incidents_with_explanations.csv", mime="text/csv")

st.success("Done.")
