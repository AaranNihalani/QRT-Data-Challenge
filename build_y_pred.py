import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

# survival libs
from lifelines import CoxPHFitter

# xgboost
import xgboost as xgb

# sksurv for RSF
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except Exception:
    SKSURV_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Config & utils
# -----------------------------

DEFAULT_PENALIZER_GRID = [0.01, 0.1, 0.5, 1.0, 2.0]


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# -----------------------------
# Data loading
# -----------------------------

def load_data(data_dir: str):
    train_y = read_csv_safe(os.path.join(data_dir, "Y_train.csv"))
    clin_train = read_csv_safe(os.path.join(data_dir, "X_train", "clinical_train.csv"))
    mol_train = read_csv_safe(os.path.join(data_dir, "X_train", "molecular_train.csv"))
    clin_test = read_csv_safe(os.path.join(data_dir, "X_test", "clinical_test.csv"))
    mol_test = read_csv_safe(os.path.join(data_dir, "X_test", "molecular_test.csv"))

    # drop missing labels
    train_y = train_y.dropna(subset=["OS_YEARS", "OS_STATUS"]) 
    valid_ids = set(train_y["ID"].astype(str))
    clin_train = clin_train[clin_train["ID"].astype(str).isin(valid_ids)].copy()
    mol_train = mol_train[mol_train["ID"].astype(str).isin(valid_ids)].copy()

    return clin_train, mol_train, clin_test, mol_test, train_y


# -----------------------------
# Molecular feature engineering (expanded)
# -----------------------------

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def aggregate_molecular(mol_df: pd.DataFrame, top_n_genes: int = 200, driver_genes: Optional[List[str]] = None,
                        gene2path: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    # Ensure VAF numeric
    if "VAF" in mol_df.columns:
        mol_df["VAF"] = pd.to_numeric(mol_df["VAF"], errors="coerce").fillna(0.0)

    mol_df["GENE"] = mol_df["GENE"].astype(str)
    mol_df["EFFECT"] = mol_df.get("EFFECT", pd.Series(index=mol_df.index, data=np.nan)).astype(str)

    # Base aggregates
    base = mol_df.groupby("ID").agg(
        mut_count=("GENE", "count"),
        vaf_mean=("VAF", "mean"),
        vaf_max=("VAF", "max"),
        vaf_sum=("VAF", "sum"),
    )

    # VAF distribution per sample
    vaf_stats = mol_df.groupby("ID")["VAF"].agg(["median", "std", lambda x: x.skew() if hasattr(x, 'skew') else 0.0])
    vaf_stats.columns = ["vaf_median", "vaf_std", "vaf_skew"]

    # High VAF counts
    high_vaf = mol_df.assign(high_vaf=(mol_df["VAF"] > 0.2).astype(int)).groupby("ID")["high_vaf"].sum()

    # Effect counts
    effect_counts = mol_df.groupby(["ID", "EFFECT"]).size().unstack(fill_value=0)
    effect_counts.columns = [f"effect_{safe_name(c)}" for c in effect_counts.columns]

    # Top genes presence & vaf sums
    top_genes = mol_df["GENE"].value_counts().head(top_n_genes).index.tolist()
    mol_top = mol_df[mol_df["GENE"].isin(top_genes)].copy()

    gene_presence = (
        mol_top.assign(val=1)
        .pivot_table(index="ID", columns="GENE", values="val", aggfunc="max", fill_value=0)
    )
    gene_presence.columns = [f"gene_{safe_name(g)}_present" for g in gene_presence.columns]

    gene_vaf_sum = (
        mol_top.pivot_table(index="ID", columns="GENE", values="VAF", aggfunc="sum", fill_value=0)
    )
    gene_vaf_sum.columns = [f"gene_{safe_name(g)}_vaf_sum" for g in gene_vaf_sum.columns]

    # Driver gene flags (if provided)
    driver_df = pd.DataFrame(index=mol_df["ID"].unique())
    if driver_genes:
        drv = (
            mol_df[mol_df["GENE"].isin(driver_genes)].assign(val=1)
            .pivot_table(index="ID", columns="GENE", values="val", aggfunc="max", fill_value=0)
        )
        drv.columns = [f"driver_{safe_name(g)}" for g in drv.columns]
        driver_df = drv

    # Pathway aggregation (if gene2path mapping provided)
    pathway_counts = pd.DataFrame(index=mol_df["ID"].unique())
    if gene2path:
        # expand gene2path into per-row PATHWAYS list
        mol_df = mol_df.copy()
        mol_df["PATHWAYS"] = mol_df["GENE"].map(lambda g: gene2path.get(g, []))
        rows = []
        for idx, row in mol_df.iterrows():
            for p in row["PATHWAYS"]:
                rows.append((row["ID"], p, row.get("VAF", 0.0)))
        if rows:
            dfp = pd.DataFrame(rows, columns=["ID", "PATHWAY", "VAF"]) 
            pc = dfp.groupby(["ID", "PATHWAY"]).agg(cnt=("VAF", "size"), vaf_sum=("VAF", "sum")).unstack(fill_value=0)
            # flatten
            pc.columns = [f"path_{safe_name(c[1])}_{agg}" for c, agg in zip(pc.columns, ["cnt"]*len(pc.columns))]
            pathway_counts = pc

    # Combine
    pieces = [base, vaf_stats, high_vaf.rename("high_vaf_count"), effect_counts, gene_presence, gene_vaf_sum, driver_df, pathway_counts]
    agg = pieces[0].join([p for p in pieces[1:] if p is not None], how="left")
    agg = agg.fillna(0)
    agg.index = agg.index.astype(str)
    return agg


# -----------------------------
# Clinical preprocessing
# -----------------------------

def build_clinical_preprocessor(clin_df: pd.DataFrame):
    num_cols = [c for c in ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"] if c in clin_df.columns]
    cat_cols = [c for c in ["CENTER"] if c in clin_df.columns]
    text_col = "CYTOGENETICS" if "CYTOGENETICS" in clin_df.columns else None

    # log-transform a few counts
    for col in ["WBC", "ANC", "MONOCYTES", "PLT"]:
        if col in clin_df.columns:
            clin_df[col] = np.log1p(clin_df[col].astype(float))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))
    if text_col:
        text_pipeline = Pipeline([
            ("fill", FunctionTransformer(lambda s: s.fillna("").astype(str), validate=False)),
            ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9\-\+]+", ngram_range=(1,2), max_features=400)),
        ])
        transformers.append(("txt", text_pipeline, text_col))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.1)
    return clin_df, pre


# -----------------------------
# Feature matrix
# -----------------------------

def build_feature_matrix(clin_df: pd.DataFrame, mol_df: pd.DataFrame, preprocessor: ColumnTransformer,
                         top_n_genes: int = 200, driver_genes: Optional[List[str]] = None, gene2path: Optional[Dict[str, List[str]]] = None):
    # Aggregate molecular
    mol_agg = aggregate_molecular(mol_df, top_n_genes=top_n_genes, driver_genes=driver_genes, gene2path=gene2path)

    # Fit/transform clinical
    ids = clin_df["ID"].astype(str).values
    X_clin = preprocessor.fit_transform(clin_df)
    if hasattr(X_clin, "toarray"):
        X_clin = X_clin.toarray()

    # Build column names
    clin_feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            clin_feature_names.extend([f"num_{c}" for c in cols])
        elif name == "cat":
            ohe = trans.named_steps.get("onehot")
            if ohe is not None:
                cats = ohe.get_feature_names_out(cols)
                clin_feature_names.extend([f"cat_{c}" for c in cats])
        elif name == "txt":
            tfidf = trans.named_steps.get("tfidf") if hasattr(trans, "named_steps") else None
            if tfidf is not None and hasattr(tfidf, "get_feature_names_out"):
                tf_names = tfidf.get_feature_names_out()
                clin_feature_names.extend([f"txt_{t}" for t in tf_names])

    X_clin_df = pd.DataFrame(X_clin, index=ids, columns=clin_feature_names)
    X = X_clin_df.join(mol_agg, how="left")
    X = X.fillna(0)
    X.index = X.index.astype(str)
    return X


# -----------------------------
# Align and clean
# -----------------------------

def align_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train_a = X_train.reindex(columns=all_cols, fill_value=0)
    X_test_a = X_test.reindex(columns=all_cols, fill_value=0)
    return X_train_a, X_test_a


def drop_zero_variance(X_train: pd.DataFrame, X_test: pd.DataFrame):
    zero_var = [c for c in X_train.columns if float(X_train[c].std()) == 0.0]
    if zero_var:
        X_train = X_train.drop(columns=zero_var)
        X_test = X_test.drop(columns=zero_var, errors="ignore")
    return X_train, X_test


# -----------------------------
# Models: Cox (lifelines), RSF, XGBoost, DeepSurv
# -----------------------------

def fit_elastic_cox(X: pd.DataFrame, y_df: pd.DataFrame, penalizer: float = 0.1, l1_ratio: float = 0.5) -> CoxPHFitter:
    df = X.copy()
    df["duration"] = y_df.set_index("ID").loc[df.index, "OS_YEARS"].values
    df["event"] = y_df.set_index("ID").loc[df.index, "OS_STATUS"].values.astype(int)

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df, duration_col="duration", event_col="event", show_progress=False)
    return cph


def oof_predictions_kfold(model_name: str, X: pd.DataFrame, y_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """Return OOF training predictions and test predictions using KFold for a given model_name in {cox, rsf, xgb, deepsurv}"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))

    # survival structured array for sksurv
    if SKSURV_AVAILABLE:
        y_struct = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_df.set_index("ID").loc[X.index])
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{n_splits} for {model_name}")
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr = y_df.set_index("ID").loc[X_tr.index]
        y_val = y_df.set_index("ID").loc[X_val.index]

        if model_name == "cox":
            cph = fit_elastic_cox(X_tr, y_tr, penalizer=0.1, l1_ratio=0.5)
            val_score = cph.predict_partial_hazard(X_val).values.flatten()
            oof[val_idx] = val_score

        elif model_name == "rsf":
            if not SKSURV_AVAILABLE:
                raise RuntimeError("sksurv not available; install scikit-survival to use RSF")
            rsf = RandomSurvivalForest(n_estimators=400, min_samples_leaf=8, max_features="sqrt", n_jobs=-1, random_state=42)
            y_tr_struct = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_tr)
            rsf.fit(X_tr.values, y_tr_struct)
            # predict risk: use predict_survival_function and integrate or use predict
            val_risk = -rsf.predict(X_val.values)  # negative so higher -> worse
            oof[val_idx] = val_risk

        elif model_name == "xgb":
            # XGBoost survival: train with survival:cox objective; label is duration; events handled via sample weight trick
            dtrain = xgb.DMatrix(X_tr.values, label=y_tr["OS_YEARS"].values)
            dval = xgb.DMatrix(X_val.values, label=y_val["OS_YEARS"].values)
            params = {"objective": "survival:cox", "eval_metric": "cox-nloglik", "eta": 0.03, "max_depth": 6, "subsample": 0.8}
            # Try to use GPU if XGBoost build supports it (note: on macOS GPU support is limited)
            try:
                params.update({"tree_method": "hist", "device": "cpu"})
                print("Using XGBoost CPU mode with tree_method='hist'")
            except Exception:
                print("Could not set GPU params for XGBoost; will attempt training and fallback to CPU if it fails")

            # Train with a fallback to CPU if GPU-backed train fails
            try:
                bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
            except Exception as e:
                print("XGBoost GPU training failed, retrying on CPU. Error:", e)
                params_cpu = params.copy()
                params_cpu.update({"tree_method": "hist", "device": "cpu"})
                bst = xgb.train(params_cpu, dtrain, num_boost_round=1000, evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)

            val_risk = bst.predict(dval)
            oof[val_idx] = val_risk

        elif model_name == "deepsurv":
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available for DeepSurv")

            # CPU only
            model = DeepSurvModel(input_dim=X_tr.shape[1])
            model = train_deepsurv(
                model,
                X_tr.values.astype(np.float32),
                y_tr,
                X_val.values.astype(np.float32),
                y_val,
                device=torch.device("cpu"),  # force CPU
                epochs=5,                    # small for test
                batch_size=16,
                patience=2,
                verbose=True
            )
            val_risk = predict_deepsurv_risk(model, X_val.values.astype(np.float32), device=torch.device("cpu"))
            oof[val_idx] = val_risk


        else:
            raise ValueError("Unknown model_name")

    return oof

# -----------------------------
# Stacking & ensemble
# -----------------------------

def stack_and_blend(X_train, X_test, y_train, models_oof_preds: Dict[str, np.ndarray], test_preds_dict: Dict[str, np.ndarray]):
    """Given OOF preds (train) and test preds (per-model), fit a meta-model (elastic-net Cox) on OOF preds.
    Returns blended test prediction array.
    """
    # Build meta X
    meta_X = np.vstack([models_oof_preds[m] for m in models_oof_preds]).T
    meta_X_test = np.vstack([test_preds_dict[m] for m in test_preds_dict]).T

    meta_cols = [f"m_{safe_name(m)}" for m in models_oof_preds.keys()]
    meta_df = pd.DataFrame(meta_X, index=X_train.index, columns=meta_cols)
    meta_df_test = pd.DataFrame(meta_X_test, index=X_test.index, columns=meta_cols)

    # Fit small Cox on meta features
    meta_y = y_train.copy()
    cph_meta = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    meta_df_fit = meta_df.copy()
    meta_df_fit["duration"] = meta_y.set_index("ID").loc[meta_df_fit.index, "OS_YEARS"].values
    meta_df_fit["event"] = meta_y.set_index("ID").loc[meta_df_fit.index, "OS_STATUS"].values.astype(int)
    cph_meta.fit(meta_df_fit, duration_col="duration", event_col="event", show_progress=False)

    blended_test = cph_meta.predict_partial_hazard(meta_df_test).values.flatten()
    return blended_test


# -----------------------------
# Runner / main
# -----------------------------

def main(args):
    clin_train, mol_train, clin_test, mol_test, y_train = load_data(args.data_dir)

    # optional: load driver genes and gene2path mapping if provided
    driver_genes = None
    gene2path = None
    if args.driver_genes:
        driver_genes = [g.strip() for g in open(args.driver_genes).read().splitlines() if g.strip()]
    if args.gene2path:
        gene2path = json.load(open(args.gene2path))

    clin_train_proc, preprocessor = build_clinical_preprocessor(clin_train.copy())
    X_train = build_feature_matrix(clin_train_proc, mol_train, preprocessor, top_n_genes=args.top_genes, driver_genes=driver_genes, gene2path=gene2path)

    # prepare test using fitted preprocessor
    ids_test = clin_test["ID"].astype(str).values
    X_clin_test = preprocessor.transform(clin_test)
    if hasattr(X_clin_test, "toarray"):
        X_clin_test = X_clin_test.toarray()

    # reconstruct feature names
    clin_feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            clin_feature_names.extend([f"num_{c}" for c in cols])
        elif name == "cat":
            ohe = trans.named_steps.get("onehot")
            if ohe is not None:
                cats = ohe.get_feature_names_out(cols)
                clin_feature_names.extend([f"cat_{c}" for c in cats])
        elif name == "txt":
            tfidf = trans.named_steps.get("tfidf") if hasattr(trans, "named_steps") else None
            if tfidf is not None and hasattr(tfidf, "get_feature_names_out"):
                tf_names = tfidf.get_feature_names_out()
                clin_feature_names.extend([f"txt_{t}" for t in tf_names])

    X_clin_test_df = pd.DataFrame(X_clin_test, index=ids_test, columns=clin_feature_names)
    mol_test_agg = aggregate_molecular(mol_test, top_n_genes=args.top_genes, driver_genes=driver_genes, gene2path=gene2path)
    X_test = X_clin_test_df.join(mol_test_agg, how="left").fillna(0)

    # Align
    X_train_al, X_test_al = align_train_test(X_train, X_test)
    X_train_al, X_test_al = drop_zero_variance(X_train_al, X_test_al)

    # Ensure indices are strings
    X_train_al.index = X_train_al.index.astype(str)
    X_test_al.index = X_test_al.index.astype(str)

    # Fit baseline Cox (full data) for a quick baseline
    print("Fitting baseline Cox (elastic-net) on full training data...")
    try:
        cph_full = fit_elastic_cox(X_train_al, y_train, penalizer=0.1, l1_ratio=0.5)
        cph_test_risk = cph_full.predict_partial_hazard(X_test_al).values.flatten()
    except Exception as e:
        print("Cox failed:", e)
        cph_test_risk = np.zeros(len(X_test_al))

    # OOF for RSF, XGB, DeepSurv (if available)
    oof_preds = {}
    test_preds = {}

    # RSF
    if SKSURV_AVAILABLE:
        print("Generating OOF for RSF...")
        rsf_oof = oof_predictions_kfold("rsf", X_train_al, y_train, n_splits=5)
        oof_preds["rsf"] = rsf_oof
        # fit full RSF and predict test
        rsf_full = RandomSurvivalForest(n_estimators=800, min_samples_leaf=8, max_features="sqrt", n_jobs=-1, random_state=42)
        y_struct_full = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_train.set_index("ID").loc[X_train_al.index])
        rsf_full.fit(X_train_al.values, y_struct_full)
        test_preds["rsf"] = -rsf_full.predict(X_test_al.values)
    else:
        print("sksurv not available: skipping RSF")

    # XGBoost
    try:
        print("Generating OOF for XGBoost...")
        xgb_oof = oof_predictions_kfold("xgb", X_train_al, y_train, n_splits=5)
        oof_preds["xgb"] = xgb_oof
        # full model
        dtrain = xgb.DMatrix(X_train_al.values, label=y_train.set_index("ID").loc[X_train_al.index, "OS_YEARS"].values)
        dtest = xgb.DMatrix(X_test_al.values)
        params = {"objective": "survival:cox", "eval_metric": "cox-nloglik", "eta": 0.03, "max_depth": 6, "subsample": 0.8}
        try:
            params.update({"tree_method": "hist", "device": "cpu"})
            print("Attempting to use XGBoost CPU (hist) with device='cpu'")
        except Exception:
            print("Could not set CPU params for XGBoost; will attempt CPU if training fails")
        try:
            bst = xgb.train(params, dtrain, num_boost_round=1000, verbose_eval=False)
        except Exception as e:
            print("XGBoost GPU training failed on full data, retrying on CPU. Error:", e)
            params_cpu = params.copy()
            params_cpu.update({"tree_method": "hist", "device": "cpu"})
            bst = xgb.train(params_cpu, dtrain, num_boost_round=1000, verbose_eval=False)
        test_preds["xgb"] = bst.predict(dtest)
    except Exception as e:
        print("XGBoost training failed:", e)


    # Cox baseline in blend
    oof_preds["cox"] = cph_full.predict_partial_hazard(X_train_al).values.flatten()
    test_preds["cox"] = cph_test_risk

    # Ensure keys consistent
    for k in list(test_preds.keys()):
        if len(test_preds[k]) != len(X_test_al):
            print(f"Warning: test_preds length mismatch for {k}")

    # Report CV concordance (c-index) from OOF predictions per model
    try:
        from lifelines.utils import concordance_index
        durations = y_train.set_index("ID").loc[X_train_al.index, "OS_YEARS"].values
        events = y_train.set_index("ID").loc[X_train_al.index, "OS_STATUS"].values.astype(int)
        for k, preds in oof_preds.items():
            try:
                ci = concordance_index(durations, preds, events)
                print(f"Estimated CV concordance ({k}): {ci:.4f}")
            except Exception as e:
                print(f"Failed to compute c-index for {k}: {e}")
        # Approximate blended concordance by training meta Cox on OOF features
        model_keys = list(oof_preds.keys())
        if model_keys:
            meta_X = np.vstack([oof_preds[m] for m in model_keys]).T
            meta_cols = [f"m_{safe_name(m)}" for m in model_keys]
            meta_df = pd.DataFrame(meta_X, index=X_train_al.index, columns=meta_cols)
            cph_meta = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
            meta_df_fit = meta_df.copy()
            meta_df_fit["duration"] = durations
            meta_df_fit["event"] = events
            cph_meta.fit(meta_df_fit, duration_col="duration", event_col="event", show_progress=False)
            meta_risk = cph_meta.predict_partial_hazard(meta_df).values.flatten()
            ci_blend = concordance_index(durations, meta_risk, events)
            print(f"Estimated CV concordance (stacked blend, approx): {ci_blend:.4f}")
        else:
            print("No OOF predictions available to estimate blended concordance.")
    except Exception as e:
        print("Failed to estimate concordance:", e)

    # Simple normalized average blending as fallback
    # Normalize OOF per-model, build meta-model via stacking
    print("Stacking and blending models...")
    blended = stack_and_blend(X_train_al, X_test_al, y_train, oof_preds, test_preds)

    # Prepare submission
    submission = pd.DataFrame({"ID": X_test_al.index, "risk_score": blended})
    submission = submission.sort_values("ID")
    out_path = args.out
    submission.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}. Rows: {len(submission)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="path to data directory containing X_train, X_test, Y_train.csv")
    parser.add_argument("--out", type=str, default="Y_pred.csv", help="output predictions CSV")
    parser.add_argument("--top_genes", type=int, default=200)
    parser.add_argument("--driver_genes", type=str, default=None, help="optional path to newline-separated driver genes")
    parser.add_argument("--gene2path", type=str, default=None, help="optional path to gene->pathway JSON mapping")
    args = parser.parse_args()

    main(args)
