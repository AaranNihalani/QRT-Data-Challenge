import argparse
import json
import os
import re
import time
import logging
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

from lifelines.utils import concordance_index

# PyTorch (DeepSurv)
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
# Logging utilities
# -----------------------------

def setup_logger(debug: bool = True, logfile: Optional[str] = None):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(logfile)] if logfile else [])
    )
    logging.info("Logger initialized. Debug=%s", debug)


# -----------------------------
# Data I/O
# -----------------------------

def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_data(data_dir: str):
    clin_train = read_csv_safe(os.path.join(data_dir, "X_train", "clinical_train.csv"))
    mol_train = read_csv_safe(os.path.join(data_dir, "X_train", "molecular_train.csv"))
    clin_test = read_csv_safe(os.path.join(data_dir, "X_test", "clinical_test.csv"))
    mol_test = read_csv_safe(os.path.join(data_dir, "X_test", "molecular_test.csv"))
    y_train = read_csv_safe(os.path.join(data_dir, "Y_train.csv"))
    return clin_train, mol_train, clin_test, mol_test, y_train


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

    # Pathway aggregates (optional)
    pathway_counts = None
    if gene2path is not None and "GENE" in mol_df.columns:
        dfp = mol_df.copy()
        dfp["PATHWAY"] = dfp["GENE"].map(lambda g: "|".join(gene2path.get(str(g), [])))
        dfp = dfp[dfp["PATHWAY"].astype(str) != ""]
        if not dfp.empty:
            pc = dfp.groupby(["ID", "PATHWAY"]).agg(cnt=("VAF", "size"), vaf_sum=("VAF", "sum")).unstack(fill_value=0)
            pc.columns = [f"path_{safe_name(c[1])}_{agg}" for c, agg in zip(pc.columns, ["cnt"]*len(pc.columns))]
            pathway_counts = pc

    pieces = [base, vaf_stats, high_vaf.rename("high_vaf_count"), effect_counts, gene_presence, gene_vaf_sum, diversity, pathway_counts]
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
        # collapse rare categories in-place
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
# Feature matrix & alignment
# -----------------------------

def build_feature_matrix(clin_df: pd.DataFrame, mol_df: pd.DataFrame, preprocessor: ColumnTransformer,
                         top_n_genes: int = 200, driver_genes: Optional[List[str]] = None, gene2path: Optional[Dict[str, List[str]]] = None):
    mol_agg = aggregate_molecular(mol_df, top_n_genes=top_n_genes, driver_genes=driver_genes, gene2path=gene2path)
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
    X = X_clin_df.join(mol_agg, how="left").fillna(0)
    X.index = X.index.astype(str)
    return X


def align_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train_a = X_train.reindex(columns=all_cols, fill_value=0)
    X_test_a = X_test.reindex(columns=all_cols, fill_value=0)
    return X_train_a, X_test_a


def drop_zero_variance(X_train: pd.DataFrame, X_test: pd.DataFrame):
    stds = X_train.std(axis=0)
    keep = stds[stds > 1e-12].index.tolist()
    return X_train[keep].copy(), X_test[keep].copy()


def apply_molecular_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, variance: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
# DeepSurv implementation (CPU-safe) with robust debugging
# -----------------------------

class DeepSurvModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # linear predictor
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepSurvLoss(nn.Module):
    """Breslow partial log-likelihood loss, vectorized with log-sum-exp for numerical stability."""
    def forward(self, risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        # sort by durations ascending so risk set is suffix
        order = torch.argsort(durations)
        t = durations[order]
        e = events[order]
        r = risk[order]
        # cumulative log-sum-exp from end to start
        # compute logsumexp over suffixes efficiently
        max_suffix = torch.cummax(r.flip(0), dim=0)[0].flip(0)
        exp_cum = torch.cumsum(torch.exp(r - max_suffix), dim=0)
        lse = torch.log(exp_cum) + max_suffix
        # contribution only where event==1
        idx = (e > 0).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return torch.zeros((), device=risk.device)
        loglik = r[idx] - lse[idx]
        return -loglik.mean()


def train_deepsurv(X_tr: np.ndarray, y_tr_df: pd.DataFrame, X_val: np.ndarray, y_val_df: pd.DataFrame,
                   epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
                   hidden_dims: List[int] = [128, 64], dropout: float = 0.1, patience: int = 20,
                   device: str = "cpu", seed: int = 42, debug: bool = True) -> nn.Module:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available for DeepSurv")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr = np.asarray(X_tr, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    # Extract durations/events (assume y_* already aligned and cleaned for NaNs)
    durations_tr = torch.from_numpy(pd.to_numeric(y_tr_df["OS_YEARS"], errors="coerce").values.astype(np.float32))
    events_tr = torch.from_numpy(pd.to_numeric(y_tr_df["OS_STATUS"], errors="coerce").values.astype(np.int64))
    durations_val = torch.from_numpy(pd.to_numeric(y_val_df["OS_YEARS"], errors="coerce").values.astype(np.float32))
    events_val = torch.from_numpy(pd.to_numeric(y_val_df["OS_STATUS"], errors="coerce").values.astype(np.int64))

    # Debug info
    logging.info("DeepSurv train shapes: X_tr=%s, X_val=%s", X_tr.shape, X_val.shape)
    logging.info("Durations range: train=[%.4f, %.4f], val=[%.4f, %.4f]",
                 float(durations_tr.min()), float(durations_tr.max()), float(durations_val.min()), float(durations_val.max()))
    logging.info("Events count: train=%d, val=%d", int(events_tr.sum().item()), int(events_val.sum().item()))
    logging.info("Torch version=%s, device=%s, threads=%d", torch.__version__, device, torch.get_num_threads())

    model = DeepSurvModel(input_dim=X_tr.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    model.to(device)
    criterion = DeepSurvLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    durations_tr = durations_tr.to(device)
    events_tr = events_tr.to(device)
    durations_val = durations_val.to(device)
    events_val = events_val.to(device)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk_tr = model(X_tr_t)
        if not torch.isfinite(risk_tr).all():
            logging.warning("Non-finite risk predictions detected at epoch %d", ep)
        loss_tr = criterion(risk_tr, durations_tr, events_tr)
        loss_tr.backward()
        # Gradient diagnostics
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            risk_val = model(X_val_t)
            loss_val = criterion(risk_val, durations_val, events_val)
        elapsed = time.time() - start
        logging.debug("Epoch %4d | train_loss=%.6f val_loss=%.6f grad_norm=%.4f time=%.3fs",
                      ep, loss_tr.item(), loss_val.item(), total_norm, elapsed)

        # Early stopping
        if loss_val.item() + 1e-6 < best_val:
            best_val = loss_val.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info("Early stopping at epoch %d (best_val=%.6f)", ep, best_val)
                break

        # Hang detection: warn if epoch takes unusually long (>60s)
        if elapsed > 60:
            logging.warning("Epoch %d took %.1fs, potential hang detected.", ep, elapsed)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_deepsurv_risk(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return preds.astype(np.float64)


# -----------------------------
# Utilities
# -----------------------------

def _get_y_indexed(y_df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    # Return labels indexed by ID to simplify downstream alignment
    return y_df.set_index("ID").loc[index]


def _stratify_labels(y_df: pd.DataFrame, index: pd.Index, n_bins: int = 5) -> np.ndarray:
    y = y_df.set_index("ID").loc[index]
    durations = pd.to_numeric(y["OS_YEARS"], errors="coerce").values.astype(float)
    if np.isnan(durations).any():
        fill = np.nanmedian(durations)
        if np.isnan(fill):
            fill = 0.0
        durations = np.where(np.isnan(durations), fill, durations)
    bins = pd.qcut(durations, q=n_bins, labels=False, duplicates="drop")
    return bins.astype(int)


# -----------------------------
# Main runner (DeepSurv only)
# -----------------------------

def main(args):
    setup_logger(debug=args.debug, logfile=args.logfile)
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available; cannot run DeepSurv script")

    logging.info("Loading data from %s", args.data_dir)
    clin_train, mol_train, clin_test, mol_test, y_train = load_data(args.data_dir)

    # Coerce label columns to numeric and log basic stats
    for col in ("OS_YEARS", "OS_STATUS"):
        if col in y_train.columns:
            y_train[col] = pd.to_numeric(y_train[col], errors="coerce")
    logging.info("Labels: OS_YEARS nulls=%d, OS_STATUS nulls=%d",
                 int(y_train["OS_YEARS"].isna().sum()), int(y_train["OS_STATUS"].isna().sum()))

    # Optional driver genes / pathways
    driver_genes = None
    gene2path = None
    if args.driver_genes:
        driver_genes = [g.strip() for g in open(args.driver_genes).read().splitlines() if g.strip()]
    if args.gene2path:
        gene2path = json.load(open(args.gene2path))

    # Clinical preprocessing and feature matrix
    logging.info("Building clinical preprocessor and training feature matrix...")
    clin_train_proc, preprocessor = build_clinical_preprocessor(clin_train.copy())
    X_train = build_feature_matrix(clin_train_proc, mol_train, preprocessor, top_n_genes=args.top_genes, driver_genes=driver_genes, gene2path=gene2path)
    # Test clinical must undergo the same feature engineering
    logging.info("Transforming test clinical with the fitted preprocessor...")
    clin_test_proc, _ = build_clinical_preprocessor(clin_test.copy())
    X_clin_test = preprocessor.transform(clin_test_proc)
    if hasattr(X_clin_test, "toarray"):
        X_clin_test = X_clin_test.toarray()
    # Feature names reconstruction
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
    ids_test = clin_test["ID"].astype(str).values
    X_clin_test_df = pd.DataFrame(X_clin_test, index=ids_test, columns=clin_feature_names)
    mol_test_agg = aggregate_molecular(mol_test, top_n_genes=args.top_genes, driver_genes=driver_genes, gene2path=gene2path)
    X_test = X_clin_test_df.join(mol_test_agg, how="left").fillna(0)

    # Align, reduce, clean
    X_train_al, X_test_al = align_train_test(X_train, X_test)
    X_train_al, X_test_al = drop_zero_variance(X_train_al, X_test_al)
    X_train_al, X_test_al = apply_molecular_pca(X_train_al, X_test_al, variance=0.95)
    X_train_al.index = X_train_al.index.astype(str)
    X_test_al.index = X_test_al.index.astype(str)

    # OOF CV for DeepSurv only
    logging.info("Starting DeepSurv OOF CV (%d folds)...", args.n_splits)
    labels = _stratify_labels(y_train, X_train_al.index, n_bins=5)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(X_train_al), dtype=np.float64)
    durations_all = y_train.set_index("ID").loc[X_train_al.index, "OS_YEARS"].values
    events_all = y_train.set_index("ID").loc[X_train_al.index, "OS_STATUS"].values.astype(int)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_al, labels), start=1):
        logging.info("Fold %d/%d: train=%d val=%d", fold, args.n_splits, len(tr_idx), len(val_idx))
        X_tr = X_train_al.iloc[tr_idx]
        X_val = X_train_al.iloc[val_idx]
        y_tr_df = _get_y_indexed(y_train, X_tr.index)
        y_val_df = _get_y_indexed(y_train, X_val.index)

        # Drop rows with NaNs in labels and align X accordingly
        tr_mask = y_tr_df["OS_YEARS"].notna() & y_tr_df["OS_STATUS"].notna()
        val_mask = y_val_df["OS_YEARS"].notna() & y_val_df["OS_STATUS"].notna()
        dropped_tr = int((~tr_mask).sum())
        dropped_val = int((~val_mask).sum())
        if dropped_tr or dropped_val:
            logging.warning("Fold %d: dropped %d train and %d val rows due to NaN labels",
                            fold, dropped_tr, dropped_val)
        if dropped_tr:
            valid_ids_tr = y_tr_df.index[tr_mask]
            X_tr = X_tr.loc[valid_ids_tr]
            y_tr_df = y_tr_df.loc[valid_ids_tr]
        if dropped_val:
            valid_ids_val = y_val_df.index[val_mask]
            X_val = X_val.loc[valid_ids_val]
            y_val_df = y_val_df.loc[valid_ids_val]

        # Standardize features per fold to stabilize training
        scaler = StandardScaler(with_mean=True)
        X_tr_s = scaler.fit_transform(X_tr.values)
        X_val_s = scaler.transform(X_val.values)

        try:
            model = train_deepsurv(
                X_tr_s, y_tr_df, X_val_s, y_val_df,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                hidden_dims=args.hidden_dims, dropout=args.dropout, patience=args.patience,
                device="cpu", seed=args.seed, debug=args.debug,
            )
            val_pred = predict_deepsurv_risk(model, X_val_s, device="cpu")
            # Map filtered validation IDs back to oof positions
            val_pos = X_train_al.index.get_indexer(y_val_df.index)
            oof[val_pos] = val_pred
            ci = concordance_index(y_val_df["OS_YEARS"].values, val_pred, y_val_df["OS_STATUS"].values.astype(int))
            logging.info("Fold %d concordance (DeepSurv): %.4f", fold, ci)
        except Exception as e:
            logging.exception("DeepSurv training failed on fold %d: %s", fold, e)
            val_pos = X_train_al.index.get_indexer(y_val_df.index)
            oof[val_pos] = 0.0

    # Overall CV concordance (drop NaNs)
    mask_all = np.isfinite(durations_all) & np.isfinite(events_all) & np.isfinite(oof)
    if mask_all.sum() == 0:
        logging.error("No valid rows to compute CV concordance (all NaNs or invalid)")
        cv_ci = float("nan")
    else:
        cv_ci = concordance_index(durations_all[mask_all], oof[mask_all], events_all[mask_all].astype(int))
    logging.info("Estimated CV concordance (DeepSurv OOF): %.4f", cv_ci)

    # Train full DeepSurv on all data and predict test
    logging.info("Training DeepSurv on full data and generating test predictions...")
    y_full_idxed = _get_y_indexed(y_train, X_train_al.index)
    full_mask = y_full_idxed["OS_YEARS"].notna() & y_full_idxed["OS_STATUS"].notna()
    dropped_full = int((~full_mask).sum())
    if dropped_full:
        logging.warning("Full training: dropped %d rows due to NaN labels", dropped_full)
    valid_ids_full = y_full_idxed.index[full_mask]
    X_full = X_train_al.loc[valid_ids_full]
    y_full_idxed = y_full_idxed.loc[valid_ids_full]
    scaler_full = StandardScaler(with_mean=True)
    X_full_s = scaler_full.fit_transform(X_full.values)
    X_test_s = scaler_full.transform(X_test_al.values)
    model_full = train_deepsurv(
        X_full_s, y_full_idxed, X_full_s, y_full_idxed,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        hidden_dims=args.hidden_dims, dropout=args.dropout, patience=args.patience,
        device="cpu", seed=args.seed, debug=args.debug,
    )
    test_risk = predict_deepsurv_risk(model_full, X_test_s, device="cpu")

    # Prepare submission
    submission = pd.DataFrame({"ID": X_test_al.index, "risk_score": test_risk})
    submission = submission.sort_values("ID")
    submission.to_csv(args.out, index=False)
    logging.info("Saved predictions to %s. Rows: %d", args.out, len(submission))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="path to data directory containing X_train, X_test, Y_train.csv")
    parser.add_argument("--out", type=str, default="Y_pred_deepsurv.csv", help="output predictions CSV")
    parser.add_argument("--top_genes", type=int, default=200)
    parser.add_argument("--driver_genes", type=str, default=None, help="optional path to newline-separated driver genes")
    parser.add_argument("--gene2path", type=str, default=None, help="optional path to gene->pathway JSON mapping")
    # DeepSurv params
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=lambda s: [int(x) for x in s.split(',')], default=[128, 64], help="comma-separated hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="enable verbose debug logging")
    parser.add_argument("--logfile", type=str, default=None, help="optional path to write debug logs")
    args = parser.parse_args()

    main(args)