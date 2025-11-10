import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

# survival libs
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError

# xgboost
import xgboost as xgb

# sksurv for RSF
try:
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except Exception:
    SKSURV_AVAILABLE = False

# DeepSurv / PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Config & utils
# -----------------------------

DEFAULT_PENALIZER_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
DEFAULT_L1_RATIO_GRID = [0.0, 0.25, 0.5]


def _get_y_indexed(y_df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Return y_df aligned to X index, robust to presence/absence of 'ID' column."""
    if "ID" in y_df.columns:
        yd = y_df.set_index("ID")
    else:
        yd = y_df.copy()
        yd.index = yd.index.astype(str)
    return yd.loc[index]


def _get_duration_event(y_df: pd.DataFrame, index: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
    yd = _get_y_indexed(y_df, index)
    durations = yd["OS_YEARS"].values.astype(float)
    events = yd["OS_STATUS"].values.astype(int)
    return durations, events


def _stratify_labels(y_df: pd.DataFrame, index: pd.Index, n_bins: int = 5) -> np.ndarray:
    """Build stratification labels combining event and binned durations for survival-aware splits."""
    durations, events = _get_duration_event(y_df, index)
    # Bin durations into quantiles; ensure at least 1 bin
    try:
        bins = pd.qcut(durations, q=n_bins, duplicates="drop")
        bin_codes = pd.factorize(bins, sort=True)[0]
    except Exception:
        bin_codes = np.zeros_like(durations, dtype=int)
    labels = events.astype(int) * (n_bins + 1) + bin_codes
    return labels


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
    if mol_df is None or mol_df.empty:
        return pd.DataFrame()
    # Ensure VAF numeric
    if "VAF" in mol_df.columns:
        mol_df["VAF"] = pd.to_numeric(mol_df["VAF"], errors="coerce").fillna(0.0)
    if "DEPTH" in mol_df.columns:
        mol_df["DEPTH"] = pd.to_numeric(mol_df["DEPTH"], errors="coerce").fillna(0.0)

    mol_df["GENE"] = mol_df["GENE"].astype(str)
    mol_df["EFFECT"] = mol_df.get("EFFECT", pd.Series(index=mol_df.index, data=np.nan)).astype(str)
    unique_ids = mol_df["ID"].astype(str).unique()

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

    # Diversity & entropy metrics
    diversity = pd.DataFrame(index=unique_ids)
    diversity["gene_unique"] = mol_df.groupby("ID")["GENE"].nunique()
    diversity["effect_unique"] = mol_df.groupby("ID")["EFFECT"].nunique()
    if "CHR" in mol_df.columns:
        diversity["chr_unique"] = mol_df.groupby("ID")["CHR"].nunique()
    else:
        diversity["chr_unique"] = 0.0

    def _entropy(series: pd.Series) -> float:
        counts = series.value_counts()
        if counts.empty:
            return 0.0
        probs = counts / counts.sum()
        return float(-(probs * np.log(probs + 1e-12)).sum())

    gene_entropy = mol_df.groupby("ID")["GENE"].apply(_entropy).rename("gene_entropy")
    effect_entropy = mol_df.groupby("ID")["EFFECT"].apply(_entropy).rename("effect_entropy")

    # Chromosome-level hot columns (restrict to frequent chromosomes to control width)
    chr_counts = pd.DataFrame(index=unique_ids)
    if "CHR" in mol_df.columns:
        top_chr = mol_df["CHR"].astype(str).value_counts().head(12).index.tolist()
        chr_sub = mol_df[mol_df["CHR"].astype(str).isin(top_chr)].copy()
        if not chr_sub.empty:
            chr_counts = (
                chr_sub.assign(val=1)
                .pivot_table(index="ID", columns="CHR", values="val", aggfunc="sum", fill_value=0)
            )
            chr_counts.columns = [f"chr_{safe_name(c)}_count" for c in chr_counts.columns]

    # Depth statistics
    depth_stats = pd.DataFrame(index=unique_ids)
    depth_quantiles = pd.DataFrame(index=unique_ids)
    depth_weighted = pd.DataFrame(index=unique_ids)
    if "DEPTH" in mol_df.columns:
        depth_stats = mol_df.groupby("ID")["DEPTH"].agg(["mean", "median", "std", "max", "min"]).rename(
            columns={
                "mean": "depth_mean",
                "median": "depth_median",
                "std": "depth_std",
                "max": "depth_max",
                "min": "depth_min",
            }
        )
        depth_quantiles = (
            mol_df.groupby("ID")["DEPTH"].quantile([0.1, 0.5, 0.9]).unstack().rename(
                columns=lambda q: f"depth_q{int(float(q)*100)}"
            )
        )
        mol_df["VAF_DEPTH"] = mol_df["VAF"] * mol_df["DEPTH"]
        depth_weighted = mol_df.groupby("ID")["VAF_DEPTH"].agg(["sum", "mean"]).rename(
            columns={"sum": "vaf_depth_sum", "mean": "vaf_depth_mean"}
        )

    # VAF quantiles to capture clonal architecture
    if mol_df["VAF"].abs().sum() > 0:
        vaf_quantiles = (
            mol_df.groupby("ID")["VAF"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack().rename(
                columns=lambda q: f"vaf_q{int(float(q)*100)}"
            )
        )
    else:
        vaf_quantiles = pd.DataFrame(index=unique_ids)

    # Impact-driven counts
    effect_lower = mol_df["EFFECT"].str.lower()
    mol_df["is_high_impact"] = effect_lower.fillna("").str.contains(
        r"stop|frameshift|splice|nonsense|del"
    ).astype(int)
    mol_df["is_splice"] = effect_lower.fillna("").str.contains("splice").astype(int)
    if not mol_df.empty:
        impact_stats = mol_df.groupby("ID")[["is_high_impact", "is_splice"]].agg(["sum", "mean"])
        impact_stats.columns = [
            f"{col}_{stat}" for col, stat in impact_stats.columns.to_flat_index()
        ]
    else:
        impact_stats = pd.DataFrame(index=unique_ids)

    # Mutation burden normalized by gene set size
    if not diversity.empty:
        mut_counts = base["mut_count"].reindex(diversity.index).fillna(0.0)
        burden = pd.DataFrame(index=diversity.index)
        burden["mut_per_gene_unique"] = mut_counts / (diversity["gene_unique"].replace(0, np.nan))
        burden["mut_per_gene_unique"] = burden["mut_per_gene_unique"].fillna(0.0)
    else:
        burden = pd.DataFrame(index=unique_ids)

    # Driver gene flags (if provided)
    driver_df = pd.DataFrame(index=unique_ids)
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
    pieces = [
        base,
        vaf_stats,
        vaf_quantiles,
        depth_stats,
        depth_quantiles,
        depth_weighted,
        diversity,
        gene_entropy.to_frame(),
        effect_entropy.to_frame(),
        burden[["mut_per_gene_unique"]],
        high_vaf.rename("high_vaf_count"),
        impact_stats,
        chr_counts,
        effect_counts,
        gene_presence,
        gene_vaf_sum,
        driver_df,
        pathway_counts,
    ]
    agg = pieces[0].join([p for p in pieces[1:] if p is not None], how="left")
    agg = agg.fillna(0)
    agg.index = agg.index.astype(str)
    return agg


# -----------------------------
# Clinical preprocessing
# -----------------------------

def build_clinical_preprocessor(clin_df: pd.DataFrame):
    clin_df = clin_df.copy()
    base_num_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    num_cols = [c for c in base_num_cols if c in clin_df.columns]
    cat_cols = [c for c in ["CENTER"] if c in clin_df.columns]
    text_col = "CYTOGENETICS" if "CYTOGENETICS" in clin_df.columns else None

    derived_numeric_cols: List[str] = []
    log_cols = {c for c in ["WBC", "ANC", "MONOCYTES", "PLT"] if c in clin_df.columns}
    EPS = 1e-3

    def _safe_ratio(numer: str, denom: str, name: str):
        if numer in clin_df.columns and denom in clin_df.columns:
            numer_vals = pd.to_numeric(clin_df[numer], errors="coerce")
            denom_vals = pd.to_numeric(clin_df[denom], errors="coerce")
            clin_df[name] = numer_vals / (denom_vals.abs() + EPS)
            derived_numeric_cols.append(name)
            log_cols.add(name)
            return True
        return False

    def _safe_product(columns: List[str], name: str):
        for c in columns:
            if c not in clin_df.columns:
                return False
        prod = np.ones(len(clin_df))
        for c in columns:
            prod *= pd.to_numeric(clin_df[c], errors="coerce").fillna(0.0)
        clin_df[name] = prod
        derived_numeric_cols.append(name)
        log_cols.add(name)
        return True

    # Ratios capturing myeloid/lymphoid composition
    _safe_ratio("ANC", "WBC", "ratio_anc_wbc")
    _safe_ratio("MONOCYTES", "WBC", "ratio_mono_wbc")
    _safe_ratio("WBC", "PLT", "ratio_wbc_plt")
    _safe_ratio("HB", "PLT", "ratio_hb_plt")
    _safe_ratio("BM_BLAST", "WBC", "ratio_bmblast_wbc")

    # Proliferation proxy via product of blasts and WBC
    _safe_product(["BM_BLAST", "WBC"], "prod_blast_wbc")

    # Missingness fraction over core labs
    lab_cols = [c for c in base_num_cols if c in clin_df.columns]
    if lab_cols:
        clin_df["lab_missing_frac"] = clin_df[lab_cols].isna().sum(axis=1) / len(lab_cols)
        derived_numeric_cols.append("lab_missing_frac")

    # Center volume as numeric signal
    if "CENTER" in clin_df.columns:
        center_counts = clin_df["CENTER"].value_counts()
        clin_df["center_freq"] = clin_df["CENTER"].map(center_counts).astype(float)
        derived_numeric_cols.append("center_freq")
        log_cols.add("center_freq")

    # Cytogenetics text-derived numeric proxies
    if text_col:
        txt = clin_df[text_col].fillna("")
        clin_df["cyto_len"] = txt.str.len()
        clin_df["cyto_aberrations"] = txt.str.count("/")
        clin_df["cyto_symbol_count"] = txt.str.count(r"[+\-]")
        derived_numeric_cols.extend(["cyto_len", "cyto_aberrations", "cyto_symbol_count"])

    # log-transform skewed numeric columns
    for col in log_cols:
        clin_df[col] = np.log1p(pd.to_numeric(clin_df[col], errors="coerce").clip(lower=0))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    full_num_cols = sorted(set(num_cols + derived_numeric_cols))
    if full_num_cols:
        transformers.append(("num", numeric_pipeline, full_num_cols))
    if cat_cols:
        # collapse rare categories in-place to reduce OHE noise
        def _collapse_rare(df: pd.DataFrame, cols: List[str], min_count: int = 10) -> pd.DataFrame:
            df = df.copy()
            for c in cols:
                vc = df[c].astype(str).value_counts()
                rare = set(vc[vc < min_count].index.tolist())
                if rare:
                    df[c] = df[c].astype(str).apply(lambda v: "OTHER" if v in rare else v)
            return df
        clin_df = _collapse_rare(clin_df, cat_cols, min_count=10)
        transformers.append(("cat", categorical_pipeline, cat_cols))
    if text_col:
        text_pipeline = Pipeline([
            ("fill", FunctionTransformer(lambda s: s.fillna("").astype(str), validate=False)),
            ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9\-\+]+", ngram_range=(1,2), max_features=3000, min_df=2, max_df=0.9)),
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


def remove_collinearity(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop one of each highly correlated feature pair (|rho| > threshold)."""
    if X_train.shape[1] == 0:
        return X_train, X_test
    corr = X_train.corr().abs()
    # Upper triangle mask
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols, errors="ignore")
    return X_train, X_test


def apply_molecular_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, variance: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply PCA to molecular block (non num_/cat_/txt_ columns) and replace with PCs."""
    clin_prefixes = ("num_", "cat_", "txt_")
    mol_cols = [c for c in X_train.columns if not c.startswith(clin_prefixes)]
    if not mol_cols:
        return X_train, X_test
    scaler = StandardScaler(with_mean=False)
    X_train_mol = scaler.fit_transform(X_train[mol_cols])
    X_test_mol = scaler.transform(X_test[mol_cols])

    pca_full = PCA(n_components=None, svd_solver="full")
    pca_full.fit(X_train_mol)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum, variance) + 1)
    pca = PCA(n_components=k, svd_solver="full")
    Z_train = pca.fit_transform(X_train_mol)
    Z_test = pca.transform(X_test_mol)
    pc_cols = [f"mol_pca_{i+1}" for i in range(Z_train.shape[1])]
    X_train_out = X_train.drop(columns=mol_cols).copy()
    X_test_out = X_test.drop(columns=mol_cols).copy()
    X_train_out[pc_cols] = Z_train
    X_test_out[pc_cols] = Z_test
    return X_train_out, X_test_out


# -----------------------------
# Models: Cox (lifelines), RSF, XGBoost
# -----------------------------

def build_xgb_aft_dmatrix(X_slice: pd.DataFrame, y_df: pd.DataFrame) -> xgb.DMatrix:
    """Construct DMatrix with interval bounds for survival:aft objective."""
    y_idx = _get_y_indexed(y_df, X_slice.index)
    durations = y_idx["OS_YEARS"].values.astype(np.float32)
    durations = np.maximum(durations, 1e-3)
    events = y_idx["OS_STATUS"].values.astype(int)
    lower = durations.astype(np.float32)
    upper = lower.copy()
    upper[events == 0] = np.inf
    dmat = xgb.DMatrix(X_slice.values, label=durations)
    dmat.set_float_info("label_lower_bound", lower)
    dmat.set_float_info("label_upper_bound", upper)
    return dmat


def _predict_xgb_with_best_iteration(model: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
    """Predict using the best iteration/ntree if early stopping was used."""
    best_ntree = getattr(model, "best_ntree_limit", 0)
    if best_ntree and best_ntree > 0:
        return model.predict(dmat, ntree_limit=best_ntree)
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter >= 0:
        return model.predict(dmat, iteration_range=(0, best_iter + 1))
    return model.predict(dmat)

def fit_elastic_cox(X: pd.DataFrame, y_df: pd.DataFrame, penalizer: float = 0.1, l1_ratio: float = 0.5) -> CoxPHFitter:
    df = X.copy()
    durations, events = _get_duration_event(y_df, df.index)
    df["duration"] = durations
    df["event"] = events

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    try:
        cph.fit(df, duration_col="duration", event_col="event", show_progress=False)
    except ConvergenceError:
        # Retry with stronger L2 penalization
        cph = CoxPHFitter(penalizer=max(1.0, penalizer * 5.0), l1_ratio=0.0)
        cph.fit(df, duration_col="duration", event_col="event", show_progress=False)
    return cph


def tune_cox_hyperparameters(X: pd.DataFrame, y_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42,
                             penalizers: Optional[List[float]] = None, l1_ratios: Optional[List[float]] = None) -> Tuple[float, float]:
    """Grid-search Cox penalizer and l1_ratio by CV concordance."""
    if penalizers is None:
        penalizers = DEFAULT_PENALIZER_GRID
    if l1_ratios is None:
        l1_ratios = DEFAULT_L1_RATIO_GRID
    labels = _stratify_labels(y_df, X.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_cindex = -np.inf
    best_params = (penalizers[0], l1_ratios[0])
    for pen in penalizers:
        for l1 in l1_ratios:
            fold_scores = []
            for tr_idx, val_idx in skf.split(X, labels):
                X_tr = X.iloc[tr_idx]
                X_val = X.iloc[val_idx]
                durations_val, events_val = _get_duration_event(y_df, X_val.index)
                try:
                    cph = fit_elastic_cox(X_tr, y_df, penalizer=pen, l1_ratio=l1)
                    preds = cph.predict_partial_hazard(X_val).values.flatten()
                    ci = concordance_index(durations_val, preds, events_val)
                    fold_scores.append(ci)
                except Exception:
                    fold_scores.append(0.0)
            mean_ci = float(np.mean(fold_scores)) if fold_scores else -np.inf
            if mean_ci > best_cindex:
                best_cindex = mean_ci
                best_params = (pen, l1)
    return best_params


def oof_predictions_kfold(model_name: str, X: pd.DataFrame, y_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42,
                          model_params: Optional[Dict] = None):
    """Return OOF training predictions using stratified KFold for supported survival base learners."""
    labels = _stratify_labels(y_df, X.index)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    test_preds = []
    X_values = X.values
    ids = X.index.astype(str).tolist()

    # survival structured array for sksurv
    if SKSURV_AVAILABLE:
        y_struct = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_df.set_index("ID").loc[X.index])
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, labels)):
        print(f"Fold {fold+1}/{n_splits} for {model_name}")
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr = _get_y_indexed(y_df, X_tr.index)
        y_val = _get_y_indexed(y_df, X_val.index)

        if model_name == "cox":
            pen = 0.1
            l1 = 0.5
            if model_params is not None:
                pen = float(model_params.get("penalizer", pen))
                l1 = float(model_params.get("l1_ratio", l1))
            cph = fit_elastic_cox(X_tr, y_tr, penalizer=pen, l1_ratio=l1)
            # use log partial hazard to avoid exponentiation magnitudes
            val_score = cph.predict_log_partial_hazard(X_val).values.flatten()
            oof[val_idx] = val_score

        elif model_name == "rsf":
            if not SKSURV_AVAILABLE:
                raise RuntimeError("sksurv not available; install scikit-survival to use RSF")
            rsf_params = {"n_estimators": 800, "min_samples_leaf": 8, "max_features": "sqrt"}
            if model_params is not None:
                rsf_params.update(model_params)
            rsf = RandomSurvivalForest(n_estimators=rsf_params.get("n_estimators", 800),
                                       min_samples_leaf=rsf_params.get("min_samples_leaf", 8),
                                       max_features=rsf_params.get("max_features", "sqrt"),
                                       n_jobs=-1, random_state=42)
            y_tr_struct = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_tr)
            rsf.fit(X_tr.values, y_tr_struct)
            # predict risk: use predict_survival_function and integrate or use predict
            val_risk = -rsf.predict(X_val.values)  # negative so higher -> worse
            oof[val_idx] = val_risk

        elif model_name == "xgb":
            dtrain = build_xgb_aft_dmatrix(X_tr, y_df)
            dval = build_xgb_aft_dmatrix(X_val, y_df)
            params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "logistic",
                "aft_loss_distribution_scale": 1.3,
                "eta": 0.05,
                "max_depth": 6,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "min_child_weight": 4,
                "alpha": 0.1,
                "lambda": 1.0,
                "tree_method": "hist",
                "device": "cpu",
            }
            if model_params is not None:
                params.update(model_params)
            try:
                bst = xgb.train(params, dtrain, num_boost_round=1200, evals=[(dval, "val")], early_stopping_rounds=75, verbose_eval=False)
            except Exception as e:
                print("XGBoost survival training failed, retrying with conservative params. Error:", e)
                params_fallback = params.copy()
                params_fallback["eta"] = 0.03
                bst = xgb.train(params_fallback, dtrain, num_boost_round=900, evals=[(dval, "val")], early_stopping_rounds=60, verbose_eval=False)

            val_pred = _predict_xgb_with_best_iteration(bst, dval)
            oof[val_idx] = -val_pred

        elif model_name == "gbsa":
            if not SKSURV_AVAILABLE:
                raise RuntimeError("sksurv not available; cannot train GradientBoostingSurvivalAnalysis")
            gb_params = {"learning_rate": 0.05, "n_estimators": 600, "max_depth": 2, "subsample": 0.9}
            if model_params is not None:
                gb_params.update(model_params)
            y_tr_struct = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_tr)
            gbsa = GradientBoostingSurvivalAnalysis(
                learning_rate=gb_params.get("learning_rate", 0.05),
                n_estimators=gb_params.get("n_estimators", 600),
                max_depth=gb_params.get("max_depth", 2),
                subsample=gb_params.get("subsample", 0.9),
                random_state=random_state,
            )
            gbsa.fit(X_tr.values, y_tr_struct)
            val_risk = -gbsa.predict(X_val.values)
            oof[val_idx] = val_risk

        else:
            raise ValueError("Unknown model_name")

    return oof


def tune_xgb_survival(X: pd.DataFrame, y_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42, n_trials: int = 18) -> Dict:
    """Randomized search over XGBoost survival (AFT) params using CV concordance."""
    rng = np.random.default_rng(random_state)
    labels = _stratify_labels(y_df, X.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "logistic",
        "aft_loss_distribution_scale": 1.3,
        "tree_method": "hist",
        "device": "cpu",
    }
    best_ci = -np.inf
    best_params = {"eta": 0.05, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.8, "min_child_weight": 4, "alpha": 0.1, "lambda": 1.0}
    for _ in range(n_trials):
        params = best_params.copy()
        params.update({
            "eta": float(rng.uniform(0.02, 0.12)),
            "max_depth": int(rng.integers(3, 8)),
            "min_child_weight": int(rng.integers(1, 9)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "alpha": float(rng.uniform(0.0, 3.0)),
            "lambda": float(rng.uniform(0.5, 5.0)),
        })
        fold_scores = []
        for tr_idx, val_idx in skf.split(X, labels):
            X_tr = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            dtrain = build_xgb_aft_dmatrix(X_tr, y_df)
            dval = build_xgb_aft_dmatrix(X_val, y_df)
            full_params = base.copy()
            full_params.update(params)
            try:
                bst = xgb.train(full_params, dtrain, num_boost_round=700, evals=[(dval, "val")], early_stopping_rounds=60, verbose_eval=False)
                preds = _predict_xgb_with_best_iteration(bst, dval)
                durations_val, events_val = _get_duration_event(y_df, X_val.index)
                ci = concordance_index(durations_val, -preds, events_val)
                fold_scores.append(ci)
            except Exception:
                fold_scores.append(0.0)
        mean_ci = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if mean_ci > best_ci:
            best_ci = mean_ci
            best_params = params.copy()
    best_params.update(base)
    return best_params


def tune_rsf_hyperparameters(X: pd.DataFrame, y_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> Dict:
    """Simple grid search for RSF hyperparameters (if sksurv available)."""
    if not SKSURV_AVAILABLE:
        return {}
    labels = _stratify_labels(y_df, X.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = [
        {"n_estimators": 600, "min_samples_leaf": 6, "max_features": "sqrt"},
        {"n_estimators": 900, "min_samples_leaf": 8, "max_features": "sqrt"},
        {"n_estimators": 1200, "min_samples_leaf": 12, "max_features": 0.7},
        {"n_estimators": 1000, "min_samples_leaf": 6, "max_features": 0.8},
        {"n_estimators": 1200, "min_samples_leaf": 10, "max_features": "sqrt"},
    ]
    best_ci = -np.inf
    best = grid[0]
    for params in grid:
        fold_scores = []
        for tr_idx, val_idx in skf.split(X, labels):
            X_tr = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_tr = _get_y_indexed(y_df, X_tr.index)
            y_val = _get_y_indexed(y_df, X_val.index)
            try:
                y_tr_struct = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_tr)
                rsf = RandomSurvivalForest(n_estimators=params["n_estimators"], min_samples_leaf=params["min_samples_leaf"],
                                           max_features=params["max_features"], n_jobs=-1, random_state=42)
                rsf.fit(X_tr.values, y_tr_struct)
                preds = -rsf.predict(X_val.values)
                durations_val, events_val = _get_duration_event(y_df, X_val.index)
                ci = concordance_index(durations_val, preds, events_val)
                fold_scores.append(ci)
            except Exception:
                fold_scores.append(0.0)
        mean_ci = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if mean_ci > best_ci:
            best_ci = mean_ci
            best = params
    return best


def tune_gbsa_hyperparameters(X: pd.DataFrame, y_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> Dict:
    """Small grid search for GradientBoostingSurvivalAnalysis."""
    if not SKSURV_AVAILABLE:
        return {}
    labels = _stratify_labels(y_df, X.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = [
        {"learning_rate": 0.03, "n_estimators": 800, "max_depth": 2, "subsample": 0.9},
        {"learning_rate": 0.05, "n_estimators": 600, "max_depth": 2, "subsample": 0.8},
        {"learning_rate": 0.08, "n_estimators": 400, "max_depth": 3, "subsample": 0.7},
        {"learning_rate": 0.04, "n_estimators": 700, "max_depth": 3, "subsample": 0.85},
        {"learning_rate": 0.06, "n_estimators": 500, "max_depth": 2, "subsample": 0.75},
    ]
    best_ci = -np.inf
    best = grid[0]
    for params in grid:
        fold_scores = []
        for tr_idx, val_idx in skf.split(X, labels):
            X_tr = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_tr = _get_y_indexed(y_df, X_tr.index)
            try:
                y_tr_struct = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_tr)
                gbsa = GradientBoostingSurvivalAnalysis(
                    learning_rate=params["learning_rate"],
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    subsample=params["subsample"],
                    random_state=random_state,
                )
                gbsa.fit(X_tr.values, y_tr_struct)
                preds = -gbsa.predict(X_val.values)
                durations_val, events_val = _get_duration_event(y_df, X_val.index)
                ci = concordance_index(durations_val, preds, events_val)
                fold_scores.append(ci)
            except Exception:
                fold_scores.append(0.0)
        mean_ci = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if mean_ci > best_ci:
            best_ci = mean_ci
            best = params
    return best


# DeepSurv implementation removed per user request


# -----------------------------
# Stacking & ensemble
# -----------------------------

def stack_and_blend(X_train, X_test, y_train, models_oof_preds: Dict[str, np.ndarray], test_preds_dict: Dict[str, np.ndarray]):
    """Fit a meta-model (elastic-net Cox) on z-scored OOF preds and blend test predictions."""
    model_keys = [k for k in models_oof_preds.keys() if k in test_preds_dict]
    if not model_keys:
        raise ValueError("No overlapping models between OOF and test predictions for stacking.")
    missing = set(models_oof_preds.keys()).symmetric_difference(set(test_preds_dict.keys()))
    if missing:
        print(f"Warning: dropping models without both OOF and test predictions: {sorted(missing)}")
    # Build meta X using consistent column order
    meta_X = np.vstack([models_oof_preds[m] for m in model_keys]).T
    meta_X_test = np.vstack([test_preds_dict[m] for m in model_keys]).T

    meta_cols = [f"m_{safe_name(m)}" for m in model_keys]

    # z-score per column to balance models
    meta_X = (meta_X - meta_X.mean(axis=0)) / (meta_X.std(axis=0) + 1e-8)
    meta_X_test = (meta_X_test - meta_X.mean(axis=0)) / (meta_X.std(axis=0) + 1e-8)

    meta_df = pd.DataFrame(meta_X, index=X_train.index, columns=meta_cols)
    meta_df_test = pd.DataFrame(meta_X_test, index=X_test.index, columns=meta_cols)

    # Fit small Cox on meta features
    meta_y = y_train.copy()
    # lightly tune meta penalizer
    best_pen, best_l1 = tune_cox_hyperparameters(meta_df, y_train, n_splits=5)
    cph_meta = CoxPHFitter(penalizer=best_pen, l1_ratio=best_l1)
    meta_df_fit = meta_df.copy()
    meta_df_fit["duration"] = meta_y.set_index("ID").loc[meta_df_fit.index, "OS_YEARS"].values
    meta_df_fit["event"] = meta_y.set_index("ID").loc[meta_df_fit.index, "OS_STATUS"].values.astype(int)
    cph_meta.fit(meta_df_fit, duration_col="duration", event_col="event", show_progress=False)

    # use log partial hazard for stable magnitudes
    blended_test = cph_meta.predict_log_partial_hazard(meta_df_test).values.flatten()
    return blended_test


def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Monotonic rank-based normalization to [0,1]; preserves concordance."""
    ranks = pd.Series(arr).rank(method="average")
    return (ranks.values - 1) / (len(arr) - 1 + 1e-8)


def stabilize_preds(preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Clip extreme values and z-score each prediction vector for numerical stability."""
    out = {}
    for k, v in preds.items():
        arr = np.asarray(v, dtype=np.float64)
        arr = np.nan_to_num(arr, neginf=0.0, posinf=0.0)
        try:
            lo, hi = np.percentile(arr, [0.1, 99.9])
        except Exception:
            lo, hi = arr.min(), arr.max()
        arr = np.clip(arr, lo, hi)
        mu = arr.mean()
        sigma = arr.std() + 1e-8
        out[k] = (arr - mu) / sigma
    return out


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
    # IMPORTANT: apply the same clinical feature engineering to test before transform
    clin_test_proc, _ = build_clinical_preprocessor(clin_test.copy())
    X_clin_test = preprocessor.transform(clin_test_proc)
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
    # Dimensionality reduction for molecular features
    X_train_al, X_test_al = apply_molecular_pca(X_train_al, X_test_al, variance=0.95)
    # Remove high collinearity to stabilize Cox and stacking
    X_train_al, X_test_al = remove_collinearity(X_train_al, X_test_al, threshold=0.95)

    # Ensure indices are strings
    X_train_al.index = X_train_al.index.astype(str)
    X_test_al.index = X_test_al.index.astype(str)

    # Fit baseline Cox (full data) for a quick baseline
    print("Tuning Cox hyperparameters and fitting on full training data...")
    try:
        best_pen, best_l1 = tune_cox_hyperparameters(X_train_al, y_train, n_splits=5)
        cph_full = fit_elastic_cox(X_train_al, y_train, penalizer=best_pen, l1_ratio=best_l1)
        # use log partial hazard (linear predictor) for stable magnitudes
        cph_test_risk = cph_full.predict_log_partial_hazard(X_test_al).values.flatten()
    except Exception as e:
        print("Cox failed:", e)
        cph_full = fit_elastic_cox(X_train_al, y_train, penalizer=1.0, l1_ratio=0.0)
        cph_test_risk = cph_full.predict_log_partial_hazard(X_test_al).values.flatten()

    # OOF for RSF, XGB
    oof_preds = {}
    test_preds = {}

    # sksurv models (RSF + Gradient Boosting)
    if SKSURV_AVAILABLE:
        y_struct_full = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_train.set_index("ID").loc[X_train_al.index])

        print("Tuning RSF and generating OOF...")
        rsf_best = tune_rsf_hyperparameters(X_train_al, y_train, n_splits=5)
        rsf_oof = oof_predictions_kfold("rsf", X_train_al, y_train, n_splits=5, model_params=rsf_best)
        oof_preds["rsf"] = rsf_oof
        rsf_full = RandomSurvivalForest(
            n_estimators=rsf_best.get("n_estimators", 900),
            min_samples_leaf=rsf_best.get("min_samples_leaf", 8),
            max_features=rsf_best.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=42,
        )
        rsf_full.fit(X_train_al.values, y_struct_full)
        test_preds["rsf"] = -rsf_full.predict(X_test_al.values)

        print("Tuning Gradient Boosting Survival and generating OOF...")
        gbsa_best = tune_gbsa_hyperparameters(X_train_al, y_train, n_splits=5)
        gbsa_oof = oof_predictions_kfold("gbsa", X_train_al, y_train, n_splits=5, model_params=gbsa_best)
        oof_preds["gbsa"] = gbsa_oof
        gbsa_full = GradientBoostingSurvivalAnalysis(
            learning_rate=gbsa_best.get("learning_rate", 0.05),
            n_estimators=gbsa_best.get("n_estimators", 600),
            max_depth=gbsa_best.get("max_depth", 2),
            subsample=gbsa_best.get("subsample", 0.9),
            random_state=42,
        )
        gbsa_full.fit(X_train_al.values, y_struct_full)
        test_preds["gbsa"] = -gbsa_full.predict(X_test_al.values)
    else:
        print("sksurv not available: skipping RSF/GBSA block")

    # XGBoost
    try:
        print("Tuning XGBoost survival and generating OOF...")
        xgb_best = tune_xgb_survival(X_train_al, y_train, n_splits=5)
        xgb_oof = oof_predictions_kfold("xgb", X_train_al, y_train, n_splits=5, model_params=xgb_best)
        oof_preds["xgb"] = xgb_oof
        # full model
        dtrain = build_xgb_aft_dmatrix(X_train_al, y_train)
        dtest = xgb.DMatrix(X_test_al.values)
        params = xgb_best.copy()
        num_boost_round = int(params.pop("num_boost_round", 1100))
        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
        test_preds["xgb"] = -_predict_xgb_with_best_iteration(bst, dtest)
    except Exception as e:
        print("XGBoost training failed:", e)

    # Cox OOF with tuned hyperparameters
    try:
        cox_oof = oof_predictions_kfold("cox", X_train_al, y_train, n_splits=5, model_params={"penalizer": best_pen, "l1_ratio": best_l1})
        oof_preds["cox"] = cox_oof
    except Exception:
        oof_preds["cox"] = cph_full.predict_log_partial_hazard(X_train_al).values.flatten()
    test_preds["cox"] = cph_test_risk

    # DeepSurv block removed per user request

    # Ensure keys consistent
    for k in list(test_preds.keys()):
        if len(test_preds[k]) != len(X_test_al):
            print(f"Warning: test_preds length mismatch for {k}")

    # Estimate concordance on training using OOF predictions (per model)
    try:
        durations = y_train.set_index("ID").loc[X_train_al.index, "OS_YEARS"].values
        events = y_train.set_index("ID").loc[X_train_al.index, "OS_STATUS"].values.astype(int)
        for k, preds in oof_preds.items():
            try:
                ci = concordance_index(durations, preds, events)
                print(f"Estimated CV concordance ({k}): {ci:.4f}")
            except Exception as e:
                print(f"Failed to compute c-index for {k}: {e}")
        # Approximate blended concordance by training meta Cox on OOF features
        model_keys = sorted(set(oof_preds.keys()) & set(test_preds.keys()))
        if model_keys:
            meta_X = np.vstack([oof_preds[m] for m in model_keys]).T
            meta_X = (meta_X - meta_X.mean(axis=0)) / (meta_X.std(axis=0) + 1e-8)
            meta_cols = [f"m_{safe_name(m)}" for m in model_keys]
            meta_df = pd.DataFrame(meta_X, index=X_train_al.index, columns=meta_cols)
            best_pen, best_l1 = tune_cox_hyperparameters(meta_df, y_train, n_splits=5)
            cph_meta = CoxPHFitter(penalizer=best_pen, l1_ratio=best_l1)
            meta_df_fit = meta_df.copy()
            meta_df_fit["duration"] = durations
            meta_df_fit["event"] = events
            cph_meta.fit(meta_df_fit, duration_col="duration", event_col="event", show_progress=False)
            meta_risk = cph_meta.predict_log_partial_hazard(meta_df).values.flatten()
            ci_blend = concordance_index(durations, meta_risk, events)
            print(f"Estimated CV concordance (stacked blend, approx): {ci_blend:.4f}")
        else:
            print("No overlapping models for stacking; skipping blended concordance estimate.")
    except Exception as e:
        print("Failed to estimate concordance:", e)

    # Stabilize per-model predictions, then stack and blend
    oof_preds = stabilize_preds(oof_preds)
    test_preds = stabilize_preds(test_preds)
    print("Stacking and blending models...")
    blended = stack_and_blend(X_train_al, X_test_al, y_train, oof_preds, test_preds)
    # Normalize final risk to avoid huge magnitudes while preserving ranking
    blended = rank_normalize(blended)

    # Prepare submission
    submission = pd.DataFrame({"ID": X_test_al.index, "risk_score": blended})
    submission = submission.sort_values("ID")
    out_path = args.out
    submission.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}. Rows: {len(submission)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="path to data directory containing X_train, X_test, Y_train.csv")
    parser.add_argument("--out", type=str, default="Y_pred_large.csv", help="output predictions CSV")
    parser.add_argument("--top_genes", type=int, default=200)
    parser.add_argument("--driver_genes", type=str, default=None, help="optional path to newline-separated driver genes")
    parser.add_argument("--gene2path", type=str, default=None, help="optional path to gene->pathway JSON mapping")
    args = parser.parse_args()

    main(args)
