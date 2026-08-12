# -*- coding: utf-8 -*-
# =============================================================================
# GeneNetPred
#
# Trains GeneNetPred (GraphSAGE/GCN + optional Performer attention) alongside
# reimplemented DepGPS (4 architecture variants) and DeepDEP baselines, each
# with and without context-dependent nonsynonymous-to-synonymous (cdNS) edge
# features, across 10 random seeds. Supports multi-GPU training via
# DataParallel when available.
#
# USAGE:
#   python genenetpred.py
#
# REQUIREMENTS:
#   pip install performer-pytorch openpyxl networkx scikit-learn torch-geometric
# =============================================================================

import os, gc, sys, time, json, logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"]    = "0,1"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, GINConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from performer_pytorch import SelfAttention

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# =============================================================================
# DEVICE SETUP
# =============================================================================

if torch.cuda.is_available():
    n_gpus            = torch.cuda.device_count()
    device            = torch.device("cuda:0")
    USE_DATA_PARALLEL = n_gpus >= 2
    GPU_IDS           = list(range(n_gpus)) if USE_DATA_PARALLEL else [0]
else:
    device            = torch.device("cpu")
    USE_DATA_PARALLEL = False
    GPU_IDS           = []

# =============================================================================
# CONFIGURATION
# =============================================================================

DRIVE_FOLDER   = "/home/choudhury-t15/GeneNetPred"
OUTPUT_DIR     = f"{DRIVE_FOLDER}/results_v8"
FIGURES_DIR    = f"{DRIVE_FOLDER}/figures_v8"
LOGS_DIR       = f"{DRIVE_FOLDER}/logs"
CHECKPOINT_DIR = f"{DRIVE_FOLDER}/checkpoints_v8"

for d in [OUTPUT_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

log_path = f"{LOGS_DIR}/output_v8.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger()
log.info(f"GeneNetPred v8 started at {datetime.now()}")
log.info(f"Device: {device} | DataParallel: {USE_DATA_PARALLEL} | GPUs: {GPU_IDS}")

# Data paths
DEPMAP_PATH   = f"{DRIVE_FOLDER}/data/CRISPRGeneEffect.csv"
EXPR_PATH     = f"{DRIVE_FOLDER}/data/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
MUT_PATH      = f"{DRIVE_FOLDER}/data/OmicsSomaticMutationsMatrixDamaging.csv"
CNV_PATH      = f"{DRIVE_FOLDER}/data/OmicsCNGeneWGS.csv"
BIOGRID_PATH  = f"{DRIVE_FOLDER}/data/BIOGRID-ORGANISM-Homo_sapiens-5.0.256.tab3.txt"
MODEL_PATH    = f"{DRIVE_FOLDER}/data/Model.csv"
GENE2VEC_PATH = f"{DRIVE_FOLDER}/data/gene2vec_dim_200_iter_9.txt"
CDNS_PATH     = f"{DRIVE_FOLDER}/data/Han_et_al_2024_Supplement.xlsx"

# Splits
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# GeneNetPred architecture
EMBEDDING_DIM    = 128
GNN_HIDDEN       = 256
GNN_LAYERS       = 2
PERFORMER_BLOCKS = 2
DROPOUT          = 0.3
GENE2VEC_DIM     = 200
EDGE_FEAT_DIM    = 2

USE_POSITIONAL_EMBEDDING = True
LAPLACIAN_PE_DIM         = 16
RANDOM_WALK_PE_DIM       = 0
PE_DIM_TOTAL             = LAPLACIAN_PE_DIM + RANDOM_WALK_PE_DIM

# DepGPS architecture
DEPGPS_GMM_BINS   = 10
DEPGPS_EMBED_DIM  = 128
DEPGPS_GNN_HIDDEN = 256
DEPGPS_DROPOUT    = 0.3

# DeepDEP architecture
DEEPDEP_AE_HIDDEN      = 256
DEEPDEP_AE_LATENT      = 64
DEEPDEP_GENE_EMBED_DIM = 64
DEEPDEP_MLP_HIDDEN     = 256
DEEPDEP_DROPOUT        = 0.3

# Training - identical budget for all models
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
BATCH_SIZE       = 8           # kept small: full 1.5M-edge graph is replicated per-GPU under DataParallel
NUM_EPOCHS       = 300
ABLATION_EPOCHS  = 50
MULTISEED_EPOCHS = 300
PATIENCE         = 20
WARMUP_EPOCHS    = 15

# Loss
WEIGHT_THRESHOLD = -0.5
WEIGHT_VALUE     = 2.0

# cdNS weights (GeneNetPred only)
CDNS_WEIGHT_SCALE = 1.5
CDNS_WEIGHT_TOP_K = 500

# HP search - equal budget per variant
HP_LR_GRID      = [1e-2, 5e-3, 1e-3, 1e-4]
HP_DROPOUT_GRID = [0.2, 0.3, 0.4]
HP_LR_GRID_PERF = [5e-4, 1e-4, 5e-5]
HP_EPOCHS       = 10
HP_BATCH_SIZE   = 8             # smaller batch used only during hyperparameter sweeps

# Multi-seed
SEEDS       = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
RANDOM_SEED = 42

# -------------------------------------------------------------------------
# All DepGPS variants (paper Fig 2): name, gnn_type, use_perf, friendly label
# Each will be trained with use_cdns=False AND use_cdns=True  ? 8 runs total
# -------------------------------------------------------------------------
DEPGPS_VARIANTS = [
    # (internal_name, gnn_type, use_perf, display_label)
    ("DepGPS_SAGE_Perf",  "sage", True,  "DepGPS SAGE+Performer"),
    ("DepGPS_GIN_Perf",   "gin",  True,  "DepGPS GIN+Performer"),
    ("DepGPS_SAGE_Only",  "sage", False, "DepGPS SAGE-only"),
    ("DepGPS_Perf_Only",  "none", True,  "DepGPS Performer-only"),
]

# DeepDEP variants: name, use_cdns
DEEPDEP_VARIANTS = [
    ("DeepDEP_noCDNS", False, "DeepDEP (no cdNS)"),
    ("DeepDEP_cdNS",   True,  "DeepDEP (+cdNS)"),
]

log.info("Configuration loaded.")

# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(tag, data):
    path = f"{CHECKPOINT_DIR}/{tag}.json"
    with open(path, 'w') as f:
        json.dump(data, f)
    log.info(f"  Checkpoint saved: {tag}")

def load_checkpoint(tag):
    path = f"{CHECKPOINT_DIR}/{tag}.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        log.info(f"  Checkpoint loaded: {tag}")
        return data
    return None

def save_model_checkpoint(tag, state_dict, meta=None):
    path = f"{CHECKPOINT_DIR}/{tag}.pt"
    obj  = {"state_dict": {k: v.cpu() for k, v in state_dict.items()}}
    if meta:
        obj["meta"] = meta
    torch.save(obj, path)
    log.info(f"  Model checkpoint saved: {tag}")

def load_model_checkpoint(tag):
    path = f"{CHECKPOINT_DIR}/{tag}.pt"
    if os.path.exists(path):
        obj = torch.load(path, map_location='cpu')
        log.info(f"  Model checkpoint loaded: {tag}")
        return obj
    return None

# =============================================================================
# DATA LOADING
# =============================================================================

def load_depmap():
    log.info("Loading DepMap dependency scores...")
    df = pd.read_csv(DEPMAP_PATH, index_col=0)
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    log.info(f"  {df.shape[0]} cell lines x {df.shape[1]} genes")
    return df

def load_omics_table(path, label):
    log.info(f"Loading {label}...")
    df = pd.read_csv(path)
    df = df[df["IsDefaultEntryForModel"] == "Yes"].set_index("ModelID")
    drop = ["Unnamed: 0","SequencingID","ModelConditionID",
            "IsDefaultEntryForMC","IsDefaultEntryForModel"]
    df = df.drop(columns=drop, errors="ignore")
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    log.info(f"  {df.shape[0]} cell lines x {df.shape[1]} genes")
    return df

def load_biogrid():
    log.info("Loading BioGRID PPI network...")
    df = pd.read_csv(BIOGRID_PATH, sep="\t", low_memory=False)
    df = df[
        (df["Organism Name Interactor A"] == "Homo sapiens") &
        (df["Organism Name Interactor B"] == "Homo sapiens")
    ]
    edges = df[["Official Symbol Interactor A",
                "Official Symbol Interactor B"]].dropna()
    edges.columns = ["gene_a", "gene_b"]
    edges = edges[edges["gene_a"] != edges["gene_b"]]
    log.info(f"  {len(edges)} human PPI interactions")
    return edges

def load_gene2vec(genes):
    log.info("Loading gene2vec embeddings...")
    g2v = {}
    with open(GENE2VEC_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == GENE2VEC_DIM + 1:
                g2v[parts[0]] = np.array(parts[1:], dtype=np.float32)
    mat   = np.zeros((len(genes), GENE2VEC_DIM), dtype=np.float32)
    found = 0
    for i, g in enumerate(genes):
        if g in g2v:
            mat[i] = g2v[g]; found += 1
    log.info(f"  gene2vec coverage: {found}/{len(genes)} ({found/len(genes)*100:.1f}%)")
    return torch.tensor(mat)

def load_cdns():
    log.info("Loading cdNS gene pairs (Han et al. 2024)...")
    if not os.path.exists(CDNS_PATH):
        log.info(f"  WARNING: {CDNS_PATH} not found. Skipping cdNS features.")
        return {}, pd.DataFrame()
    s2 = pd.read_excel(CDNS_PATH, sheet_name='TableS2', header=2)
    s2 = s2[['Gene','Context genes','ED type','cdNS']].dropna()
    s2.columns = ['gene','context_gene','ed_type','cdns']
    try:
        s3 = pd.read_excel(CDNS_PATH, sheet_name='TableS3', header=2)
        s3 = s3[['Gene','Context genes','ED type','dNdS (missense)']].dropna()
        s3.columns = ['gene','context_gene','ed_type','cdns']
        s3['cdns'] = np.log1p(s3['cdns'].astype(float))
        all_pairs = pd.concat([s2, s3], ignore_index=True)
        log.info(f"  TableS2: {len(s2)} | TableS3: {len(s3)} pairs")
    except Exception as e:
        log.info(f"  TableS3 warning: {e}. Using TableS2 only.")
        all_pairs = s2
    cdns_dict = {}
    for _, row in all_pairs.iterrows():
        ga    = str(row['gene']).strip()
        gb    = str(row['context_gene']).strip()
        score = float(row['cdns'])
        is_s  = 1.0 if 'SYN' in str(row['ed_type']).upper() else 0.0
        cdns_dict[(ga, gb)] = (score, is_s)
        cdns_dict[(gb, ga)] = (score, is_s)
    log.info(f"  Total cdNS pairs: {len(all_pairs)}")
    return cdns_dict, all_pairs

log.info("=" * 55)
log.info("Loading all datasets...")
depmap           = load_depmap()
expr             = load_omics_table(EXPR_PATH, "gene expression")
mut              = load_omics_table(MUT_PATH,  "somatic mutations")
cnv              = load_omics_table(CNV_PATH,  "copy number variation")
biogrid          = load_biogrid()
model_df         = pd.read_csv(MODEL_PATH)
cdns_dict, cdns_df = load_cdns()
log.info("All datasets loaded.")

# =============================================================================
# PREPROCESSING AND ALIGNMENT
# =============================================================================

log.info("Aligning datasets...")
cell_lines = sorted(
    set(depmap.index) & set(expr.index) & set(mut.index) & set(cnv.index))
genes = sorted(
    set(depmap.columns) & set(expr.columns) & set(mut.columns) & set(cnv.columns))

log.info(f"  Shared cell lines: {len(cell_lines)}")
log.info(f"  Shared genes:      {len(genes)}")

depmap = depmap.loc[cell_lines, genes]
expr   = expr.loc[cell_lines, genes]
mut    = mut.loc[cell_lines, genes]
cnv    = cnv.loc[cell_lines, genes]

all_idx = np.arange(len(cell_lines))
train_idx, temp_idx = train_test_split(
    all_idx, test_size=1 - TRAIN_RATIO, random_state=RANDOM_SEED)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=RANDOM_SEED)
log.info(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

expr_scaler = StandardScaler()
cnv_scaler  = StandardScaler()
expr_scaler.fit(expr.fillna(0).iloc[train_idx].values)
cnv_scaler.fit(cnv.fillna(0).iloc[train_idx].values)

expr_scaled = np.clip(
    expr_scaler.transform(expr.fillna(0).values).astype(np.float32), -5, 5)
cnv_scaled  = np.clip(
    cnv_scaler.transform(cnv.fillna(0).values).astype(np.float32), -5, 5)
expr_scaled = np.nan_to_num(expr_scaled, nan=0.0)
cnv_scaled  = np.nan_to_num(cnv_scaled,  nan=0.0)
mut_values  = mut.fillna(0).values.astype(np.float32)

gene2vec_embeddings = load_gene2vec(genes)   # CPU

omics = {
    "expr": torch.tensor(expr_scaled),
    "mut":  torch.tensor(mut_values),
    "cnv":  torch.tensor(cnv_scaled),
}

dep_matrix     = depmap.values.astype(np.float32)
num_genes      = len(genes)
num_cell_lines = len(cell_lines)
gene_to_idx    = {g: i for i, g in enumerate(genes)}

log.info(f"  Genes: {num_genes} | Cell lines: {num_cell_lines}")
log.info("Preprocessing complete.")

# =============================================================================
# GRAPH CONSTRUCTION WITH cdNS EDGE FEATURES
# =============================================================================

def build_gene_graph(biogrid_edges, genes, cdns_dict):
    edge_dict = {}
    for _, row in biogrid_edges.iterrows():
        a, b = row["gene_a"], row["gene_b"]
        if a in gene_to_idx and b in gene_to_idx:
            i, j          = gene_to_idx[a], gene_to_idx[b]
            score, is_syn = cdns_dict.get((a, b), (0.0, 0.0))
            edge_dict[(i, j)] = [score, is_syn]
            edge_dict[(j, i)] = [score, is_syn]
    src   = [k[0] for k in edge_dict]
    dst   = [k[1] for k in edge_dict]
    feats = [edge_dict[k] for k in edge_dict]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(feats, dtype=torch.float)
    x          = torch.zeros((len(genes), 1), dtype=torch.float)
    cdns_edges = (edge_attr[:, 0] != 0).sum().item()
    log.info(f"  Edges with cdNS: {cdns_edges}/{edge_index.shape[1]}"
             f" ({cdns_edges/max(edge_index.shape[1],1)*100:.1f}%)")
    return Data(x=x, edge_index=edge_index), edge_attr

log.info("Building gene interaction graph...")
gene_graph, edge_attr = build_gene_graph(biogrid, genes, cdns_dict)
edge_index = gene_graph.edge_index.to(device)
edge_attr  = edge_attr.to(device)
log.info(f"  Nodes: {num_genes} | Edges: {edge_index.shape[1]}")

def compute_positional_embeddings(gene_graph, lap_dim):
    log.info("Computing Laplacian positional embeddings...")
    g = gene_graph.clone()
    g = AddLaplacianEigenvectorPE(k=lap_dim, attr_name='lap_pe')(g)
    pe = g.lap_pe.float()
    log.info(f"  PE shape: {pe.shape}")
    return pe

if USE_POSITIONAL_EMBEDDING:
    gene_pe = compute_positional_embeddings(gene_graph, LAPLACIAN_PE_DIM).cpu()
else:
    gene_pe = torch.zeros((num_genes, PE_DIM_TOTAL))

def build_cdns_gene_weights(genes, cdns_dict,
                             top_k=CDNS_WEIGHT_TOP_K, scale=CDNS_WEIGHT_SCALE):
    gene_max_cdns = np.zeros(len(genes), dtype=np.float32)
    for (ga, gb), (score, _) in cdns_dict.items():
        if ga in gene_to_idx:
            idx = gene_to_idx[ga]
            gene_max_cdns[idx] = max(gene_max_cdns[idx], abs(score))
    nonzero_mask = gene_max_cdns > 0
    n_nonzero    = int(nonzero_mask.sum())
    weights      = np.ones(len(genes), dtype=np.float32)
    if n_nonzero > 0:
        effective_top_k = min(top_k, n_nonzero)
        nonzero_vals    = gene_max_cdns[nonzero_mask]
        threshold       = np.sort(nonzero_vals)[::-1][effective_top_k - 1]
        upweight_mask   = (gene_max_cdns >= threshold) & nonzero_mask
        weights[upweight_mask] = scale
        log.info(f"  cdNS upweighted: {int(upweight_mask.sum())} genes")
    return torch.tensor(weights)

cdns_gene_weights = build_cdns_gene_weights(genes, cdns_dict)

# Per-gene cdNS scalar for DeepDEP (no graph so cdNS enters as node feature)
# Signed max-magnitude cdNS per gene, z-normalised ? shape [num_genes, 1]
def build_cdns_gene_scalar(genes, cdns_dict):
    raw = np.zeros(len(genes), dtype=np.float32)
    for (ga, gb), (score, _) in cdns_dict.items():
        if ga in gene_to_idx:
            idx = gene_to_idx[ga]
            if abs(score) > abs(raw[idx]):
                raw[idx] = score
    mu, sd = raw.mean(), raw.std() + 1e-8
    return torch.tensor(((raw - mu) / sd).astype(np.float32)).unsqueeze(1)

cdns_gene_scalar = build_cdns_gene_scalar(genes, cdns_dict)  # [G,1]

def build_gene_syn_ant_labels(genes, cdns_dict):
    labels = np.full(len(genes), -1.0, dtype=np.float32)
    for (ga, gb), (score, is_syn) in cdns_dict.items():
        if ga in gene_to_idx:
            labels[gene_to_idx[ga]] = is_syn
    return torch.tensor(labels)

gene_syn_ant_labels = build_gene_syn_ant_labels(genes, cdns_dict)
log.info("Graph construction complete.")

# =============================================================================
# DepGPS GMM TOKENISATION
# =============================================================================

log.info("Fitting GMM for DepGPS expression tokenization...")
ckpt_gmm = load_checkpoint("depgps_gmm_bins")
if ckpt_gmm is not None:
    expr_token_boundaries = np.array(ckpt_gmm["boundaries"])
    log.info(f"  GMM loaded ({len(expr_token_boundaries)} boundaries).")
else:
    train_expr_flat = expr_scaled[train_idx].flatten()
    sample_size     = min(500_000, len(train_expr_flat))
    rng             = np.random.default_rng(RANDOM_SEED)
    sample_idx      = rng.choice(len(train_expr_flat), sample_size, replace=False)
    gmm = GaussianMixture(n_components=DEPGPS_GMM_BINS, random_state=RANDOM_SEED,
                          max_iter=200, n_init=3)
    gmm.fit(train_expr_flat[sample_idx].reshape(-1, 1))
    means      = np.sort(gmm.means_.flatten())
    boundaries = [(means[i] + means[i+1]) / 2.0 for i in range(len(means)-1)]
    expr_token_boundaries = np.array(boundaries, dtype=np.float32)
    save_checkpoint("depgps_gmm_bins", {"boundaries": expr_token_boundaries.tolist()})
    log.info(f"  GMM fitted: {len(expr_token_boundaries)} boundaries")

def tokenize_expression(expr_vals_cpu):
    bins = torch.bucketize(expr_vals_cpu, torch.tensor(expr_token_boundaries))
    return bins.clamp(0, DEPGPS_GMM_BINS - 1)

def tokenize_mut(mut_vals_cpu):
    return (mut_vals_cpu > 0.5).long()

def tokenize_cnv(cnv_vals_cpu):
    thresholds = torch.tensor([-1.5, -0.3, 0.3, 1.0])
    bins = torch.bucketize(cnv_vals_cpu, thresholds)
    return bins.clamp(0, 4)

log.info("DepGPS tokenization ready.")

# =============================================================================
# DATASET AND DATALOADERS
# =============================================================================

class CellLineDataset(Dataset):
    def __init__(self, indices, dep_matrix):
        self.indices    = indices
        self.dep_matrix = dep_matrix

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cl_idx = self.indices[idx]
        scores = self.dep_matrix[cl_idx].copy()
        mask   = ~np.isnan(scores)
        scores = np.nan_to_num(scores, nan=0.0)
        return (
            torch.tensor(cl_idx, dtype=torch.long),
            torch.tensor(scores, dtype=torch.float),
            torch.tensor(mask,   dtype=torch.bool),
        )

pin_memory   = device.type == "cuda"
train_loader = DataLoader(CellLineDataset(train_idx, dep_matrix),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=pin_memory)
val_loader   = DataLoader(CellLineDataset(val_idx,   dep_matrix),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=pin_memory)
test_loader  = DataLoader(CellLineDataset(test_idx,  dep_matrix),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=pin_memory)

# Dedicated smaller-batch loader used only during hyperparameter sweeps
hp_train_loader = DataLoader(CellLineDataset(train_idx, dep_matrix),
                             batch_size=HP_BATCH_SIZE, shuffle=True,
                             num_workers=2, pin_memory=pin_memory)

log.info(f"  Train batches: {len(train_loader)} | Val: {len(val_loader)} | "
         f"Test: {len(test_loader)}")
log.info(f"  HP sweep train batches (batch_size={HP_BATCH_SIZE}): {len(hp_train_loader)}")

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

# ---------------------------------------------------------------------------
# A. GeneNetPred
# ---------------------------------------------------------------------------

class ModalEncoder(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU())
    def forward(self, x): return self.net(x)


class EdgeFeatureProjector(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.proj_syn = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.proj_ant = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, edge_attr, edge_index, num_nodes):
        score  = edge_attr[:, 0:1]
        is_syn = edge_attr[:, 1].bool()
        is_ant = (edge_attr[:, 0] != 0) & (~is_syn)
        hidden_dim = self.proj_syn[-1].out_features
        msg = torch.zeros(edge_attr.shape[0], hidden_dim, device=edge_attr.device)
        if is_syn.any(): msg[is_syn] = self.proj_syn(score[is_syn])
        if is_ant.any(): msg[is_ant] = self.proj_ant(score[is_ant])
        out = torch.zeros(num_nodes, hidden_dim, device=edge_attr.device)
        out.scatter_add_(0, edge_index.to(out.device)[1].unsqueeze(1).expand_as(msg), msg)
        return out


class AttentionWeighting(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        w = self.attn(x)
        return x * w, w


class AttentionHeatmapRecorder:
    def __init__(self, attn_module, gene_indices):
        self.attn_module  = attn_module
        self.gene_indices = set(gene_indices)
        self._buffer      = {gi: [] for gi in gene_indices}
        self._orig_fwd    = attn_module.forward
        recorder = self
        def _hooked(x):
            out, w = recorder._orig_fwd(x)
            for gi in recorder.gene_indices:
                recorder._buffer[gi].append(w[gi].detach().cpu().numpy())
            return out, w
        attn_module.forward = _hooked

    def get_mean_weights(self):
        rows = []
        for gi in sorted(self._buffer.keys()):
            arr = np.stack(self._buffer[gi])
            rows.append(arr.mean(axis=0))
        return np.stack(rows)

    def clear(self):
        for gi in self._buffer: self._buffer[gi] = []

    def detach(self):
        self.attn_module.forward = self._orig_fwd


class GeneNetPred(nn.Module):
    def __init__(self, num_genes, gnn_type="sage",
                 use_performer=False, use_edge_features=True):
        super().__init__()
        self.num_genes         = num_genes
        self.use_performer     = use_performer
        self.use_edge_features = use_edge_features

        self.expr_encoder  = ModalEncoder(EMBEDDING_DIM, DROPOUT)
        self.mut_encoder   = ModalEncoder(EMBEDDING_DIM, DROPOUT)
        self.cnv_encoder   = ModalEncoder(EMBEDDING_DIM, DROPOUT)
        self.gene2vec_proj = nn.Sequential(nn.Linear(GENE2VEC_DIM, EMBEDDING_DIM), nn.ReLU())
        self.pe_proj       = nn.Sequential(nn.Linear(PE_DIM_TOTAL, EMBEDDING_DIM), nn.ReLU())
        fused_dim          = 5 * EMBEDDING_DIM

        if use_edge_features:
            self.edge_projector = EdgeFeatureProjector(EDGE_FEAT_DIM, GNN_HIDDEN)
            self.edge_gate      = nn.Sequential(
                nn.Linear(GNN_HIDDEN * 2, GNN_HIDDEN), nn.Sigmoid())

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms  = nn.ModuleList()
        in_ch = fused_dim
        for _ in range(GNN_LAYERS):
            if gnn_type == "sage":
                self.gnn_layers.append(SAGEConv(in_ch, GNN_HIDDEN))
            else:
                self.gnn_layers.append(GCNConv(in_ch, GNN_HIDDEN))
            self.gnn_norms.append(nn.LayerNorm(GNN_HIDDEN))
            in_ch = GNN_HIDDEN

        if use_performer:
            self.performer_layers     = nn.ModuleList([
                SelfAttention(dim=GNN_HIDDEN, heads=4, causal=False)
                for _ in range(PERFORMER_BLOCKS)])
            self.performer_norms      = nn.ModuleList([
                nn.LayerNorm(GNN_HIDDEN) for _ in range(PERFORMER_BLOCKS)])
            self.performer_input_proj = nn.Linear(fused_dim, GNN_HIDDEN)

        self.residual_proj = nn.Linear(fused_dim, GNN_HIDDEN)
        self.attention     = AttentionWeighting(GNN_HIDDEN)
        self.dropout       = nn.Dropout(DROPOUT)
        self.output_head   = nn.Sequential(
            nn.Linear(GNN_HIDDEN, GNN_HIDDEN // 2), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(GNN_HIDDEN // 2, 1))
        self.register_buffer('_ei', None)
        self.register_buffer('_ea', None)
        self.register_buffer('_gpe', None)

    def set_graph(self, edge_index, edge_attr):
        self._ei = edge_index
        self._ea = edge_attr

    def set_gene_pe(self, gene_pe):
        self._gpe = gene_pe

    def forward(self, cl_indices, omics=None, edge_index=None, edge_attr=None, gene_pe=None):
        _dev    = cl_indices.device
        if omics is None:
            omics = globals()['omics']
        g2v_all = self.gene2vec_proj(gene2vec_embeddings.to(_dev))
        _gpe    = self._gpe.to(_dev) if self._gpe is not None else None
        pe_all  = (self.pe_proj(_gpe) if _gpe is not None
                   else torch.zeros_like(g2v_all))
        edge_nf = None

        _ei = self._ei.to(_dev)
        _ea = self._ea.to(_dev) if self._ea is not None else None
        if self.use_edge_features and hasattr(self, "edge_projector") and _ea is not None:
            edge_nf = self.edge_projector(_ea, _ei, self.num_genes)



        all_preds, all_attns = [], []
        for b in range(len(cl_indices)):
            cl_idx = cl_indices[b]
            node_feats = torch.cat([
                self.expr_encoder(omics["expr"][cl_idx,:].to(_dev).unsqueeze(1)),
                self.mut_encoder( omics["mut"][cl_idx,:].to(_dev).unsqueeze(1)),
                self.cnv_encoder( omics["cnv"][cl_idx,:].to(_dev).unsqueeze(1)),
                g2v_all, pe_all], dim=1)
            x = node_feats
            for i, (gl, gn) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
                xg = self.dropout(F.relu(gn(gl(x, _ei))))
                if self.use_edge_features and edge_nf is not None:
                    gate = self.edge_gate(torch.cat([xg, edge_nf], dim=1))
                    xg   = xg + gate * edge_nf
                if self.use_performer:
                    pi = self.performer_input_proj(x) if i == 0 else x
                    xp = F.relu(
                        self.performer_norms[min(i, PERFORMER_BLOCKS-1)](
                            self.performer_layers[min(i, PERFORMER_BLOCKS-1)](
                                pi.unsqueeze(0)).squeeze(0)))
                    xg = xg + xp
                res = self.residual_proj(x) if i == 0 else x
                x   = xg + res
            att, aw = self.attention(x)
            all_preds.append(self.output_head(att).squeeze(1))
            all_attns.append(aw)
        return torch.stack(all_preds, dim=0), torch.stack(all_attns, dim=0)


class GeneNetPredNoAttention(nn.Module):
    """Ablation: GeneNetPred without AttentionWeighting."""
    def __init__(self, num_genes, gnn_type="sage",
                 use_performer=False, use_edge_features=True):
        super().__init__()
        self.num_genes         = num_genes
        self.use_performer     = use_performer
        self.use_edge_features = use_edge_features

        self.expr_encoder  = ModalEncoder(EMBEDDING_DIM, DROPOUT)
        self.mut_encoder   = ModalEncoder(EMBEDDING_DIM, DROPOUT)
        self.cnv_encoder   = ModalEncoder(EMBEDDING_DIM, DROPOUT)
        self.gene2vec_proj = nn.Sequential(nn.Linear(GENE2VEC_DIM, EMBEDDING_DIM), nn.ReLU())
        self.pe_proj       = nn.Sequential(nn.Linear(PE_DIM_TOTAL, EMBEDDING_DIM), nn.ReLU())
        fused_dim          = 5 * EMBEDDING_DIM

        if use_edge_features:
            self.edge_projector = EdgeFeatureProjector(EDGE_FEAT_DIM, GNN_HIDDEN)
            self.edge_gate      = nn.Sequential(
                nn.Linear(GNN_HIDDEN * 2, GNN_HIDDEN), nn.Sigmoid())

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms  = nn.ModuleList()
        in_ch = fused_dim
        for _ in range(GNN_LAYERS):
            self.gnn_layers.append(
                SAGEConv(in_ch, GNN_HIDDEN) if gnn_type == "sage"
                else GCNConv(in_ch, GNN_HIDDEN))
            self.gnn_norms.append(nn.LayerNorm(GNN_HIDDEN))
            in_ch = GNN_HIDDEN

        if use_performer:
            self.performer_layers     = nn.ModuleList([
                SelfAttention(dim=GNN_HIDDEN, heads=4, causal=False)
                for _ in range(PERFORMER_BLOCKS)])
            self.performer_norms      = nn.ModuleList([
                nn.LayerNorm(GNN_HIDDEN) for _ in range(PERFORMER_BLOCKS)])
            self.performer_input_proj = nn.Linear(fused_dim, GNN_HIDDEN)

        self.residual_proj = nn.Linear(fused_dim, GNN_HIDDEN)
        self.dropout       = nn.Dropout(DROPOUT)
        self.output_head   = nn.Sequential(
            nn.Linear(GNN_HIDDEN, GNN_HIDDEN // 2), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(GNN_HIDDEN // 2, 1))
        self.register_buffer('_ei', None)
        self.register_buffer('_ea', None)
        self.register_buffer('_gpe', None)

    def set_graph(self, edge_index, edge_attr):
        self._ei = edge_index
        self._ea = edge_attr

    def set_gene_pe(self, gene_pe):
        self._gpe = gene_pe

    def forward(self, cl_indices, omics=None, edge_index=None, edge_attr=None, gene_pe=None):
        _dev    = cl_indices.device
        if omics is None:
            omics = globals()['omics']
        g2v_all = self.gene2vec_proj(gene2vec_embeddings.to(_dev))
        _gpe = self._gpe.to(_dev) if self._gpe is not None else None
        pe_all  = (self.pe_proj(_gpe) if _gpe is not None
                   else torch.zeros_like(g2v_all))
        edge_index = self._ei.to(_dev)
        edge_attr = self._ea.to(_dev) if self._ea is not None else None
        edge_nf = None
        if self.use_edge_features and edge_attr is not None:
            pass
        _ei = edge_index.to(_dev)
        _ea = edge_attr.to(_dev) if edge_attr is not None else None
        edge_nf = self.edge_projector(_ea, _ei, self.num_genes)
        all_preds = []
        for b in range(len(cl_indices)):
            cl_idx = cl_indices[b]
            node_feats = torch.cat([
                self.expr_encoder(omics["expr"][cl_idx,:].to(_dev).unsqueeze(1)),
                self.mut_encoder( omics["mut"][cl_idx,:].to(_dev).unsqueeze(1)),
                self.cnv_encoder( omics["cnv"][cl_idx,:].to(_dev).unsqueeze(1)),
                g2v_all, pe_all], dim=1)
            x = node_feats
            for i, (gl, gn) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
                xg = self.dropout(F.relu(gn(gl(x, _ei))))
                if self.use_edge_features and edge_nf is not None:
                    gate = self.edge_gate(torch.cat([xg, edge_nf], dim=1))
                    xg   = xg + gate * edge_nf
                if self.use_performer:
                    pi = self.performer_input_proj(x) if i == 0 else x
                    xp = F.relu(
                        self.performer_norms[min(i, PERFORMER_BLOCKS-1)](
                            self.performer_layers[min(i, PERFORMER_BLOCKS-1)](
                                pi.unsqueeze(0)).squeeze(0)))
                    xg = xg + xp
                x = xg + (self.residual_proj(x) if i == 0 else x)
            all_preds.append(self.output_head(x).squeeze(1))
        return torch.stack(all_preds, dim=0), None


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, cl_indices, omics=None, edge_index=None, edge_attr=None, gene_pe=None):
        _dev = cl_indices.device
        if omics is None:
            omics = globals()['omics']
        all_preds = []
        for b in range(len(cl_indices)):
            cl_idx = cl_indices[b]
            x = torch.stack([
                omics["expr"][cl_idx,:].to(_dev),
                omics["mut"][cl_idx,:].to(_dev),
                omics["cnv"][cl_idx,:].to(_dev)], dim=1)
            all_preds.append(self.net(x).squeeze(1))
        return torch.stack(all_preds, dim=0), None


# ---------------------------------------------------------------------------
# B. DepGPS - parametrized to cover all 4 paper variants + cdNS toggle
#
#    gnn_type in {"sage", "gin", "none"}   "none" = Performer-alone
#    use_perf in {True, False}             False  = GraphSAGE-only / GIN-only
#    use_cdns in {True, False}
#
#    When use_cdns=True AND gnn_type != "none":
#      cdNS edge signal is scatter-added onto target nodes via a learned
#      edge gate, exactly analogous to GeneNetPred's EdgeFeatureProjector.
#    When use_cdns=True AND gnn_type == "none" (Performer-alone):
#      Performer has no edge structure, so cdNS is injected as an additional
#      learned node-level embedding (same mechanism as DeepDEP cdNS below).
# ---------------------------------------------------------------------------

class DepGPSModel(nn.Module):
    def __init__(self, num_genes, gnn_type="sage", use_perf=True, use_cdns=False):
        super().__init__()
        assert gnn_type in ("sage", "gin", "none")
        self.num_genes = num_genes
        self.gnn_type  = gnn_type
        self.use_perf  = use_perf
        self.use_cdns  = use_cdns
        self.has_graph = gnn_type != "none"

        # Token embedding tables (all variants share these)
        self.expr_embed    = nn.Embedding(DEPGPS_GMM_BINS, DEPGPS_EMBED_DIM)
        self.mut_embed     = nn.Embedding(2,               DEPGPS_EMBED_DIM)
        self.cnv_embed     = nn.Embedding(5,               DEPGPS_EMBED_DIM)
        self.gene2vec_proj = nn.Sequential(
            nn.Linear(GENE2VEC_DIM, DEPGPS_EMBED_DIM), nn.ReLU())
        self.pe_proj       = nn.Sequential(
            nn.Linear(PE_DIM_TOTAL, DEPGPS_EMBED_DIM), nn.ReLU())

        # cdNS node feature for Performer-alone (no graph edges)
        if use_cdns and not self.has_graph:
            self.cdns_node_proj = nn.Sequential(
                nn.Linear(1, DEPGPS_EMBED_DIM), nn.ReLU())
            fused_dim = 6 * DEPGPS_EMBED_DIM
        else:
            self.cdns_node_proj = None
            fused_dim = 5 * DEPGPS_EMBED_DIM

        self.gnn_layers     = nn.ModuleList() if self.has_graph else None
        self.gnn_norms      = nn.ModuleList() if self.has_graph else None
        self.edge_gates     = nn.ModuleList() if (use_cdns and self.has_graph) else None
        self.perf_layers    = nn.ModuleList() if use_perf else None
        self.perf_norms     = nn.ModuleList() if use_perf else None
        self.perf_in_proj   = nn.ModuleList() if use_perf else None
        self.residual_projs = nn.ModuleList()

        in_ch = fused_dim
        for _ in range(GNN_LAYERS):
            if self.has_graph:
                if gnn_type == "sage":
                    self.gnn_layers.append(SAGEConv(in_ch, DEPGPS_GNN_HIDDEN))
                else:  # gin
                    gin_mlp = nn.Sequential(
                        nn.Linear(in_ch, DEPGPS_GNN_HIDDEN), nn.ReLU(),
                        nn.Linear(DEPGPS_GNN_HIDDEN, DEPGPS_GNN_HIDDEN))
                    self.gnn_layers.append(GINConv(gin_mlp))
                self.gnn_norms.append(nn.LayerNorm(DEPGPS_GNN_HIDDEN))
                if use_cdns:
                    # edge gate: projects [edge_feat_dim] ? hidden, then adds to node
                    self.edge_gates.append(nn.Sequential(
                        nn.Linear(EDGE_FEAT_DIM, DEPGPS_GNN_HIDDEN), nn.ReLU(),
                        nn.Linear(DEPGPS_GNN_HIDDEN, DEPGPS_GNN_HIDDEN)))
            if use_perf:
                self.perf_layers.append(
                    SelfAttention(dim=DEPGPS_GNN_HIDDEN, heads=4, causal=False))
                self.perf_norms.append(nn.LayerNorm(DEPGPS_GNN_HIDDEN))
                self.perf_in_proj.append(nn.Linear(in_ch, DEPGPS_GNN_HIDDEN))
            self.residual_projs.append(nn.Linear(in_ch, DEPGPS_GNN_HIDDEN))
            in_ch = DEPGPS_GNN_HIDDEN

        self.dropout     = nn.Dropout(DEPGPS_DROPOUT)
        self.output_head = nn.Sequential(
            nn.Linear(DEPGPS_GNN_HIDDEN, DEPGPS_GNN_HIDDEN // 2), nn.ReLU(),
            nn.Dropout(DEPGPS_DROPOUT),
            nn.Linear(DEPGPS_GNN_HIDDEN // 2, 1))
        self.register_buffer('_ei', None)
        self.register_buffer('_ea', None)
        self.register_buffer('_gpe', None)

    def set_graph(self, edge_index, edge_attr):
        self._ei = edge_index
        self._ea = edge_attr

    def set_gene_pe(self, gene_pe):
        self._gpe = gene_pe

    def forward(self, cl_indices, omics=None, edge_index=None, edge_attr=None, gene_pe=None):
        _dev = cl_indices.device
        if omics is None:
            omics = globals()['omics']
        g2v_all = self.gene2vec_proj(gene2vec_embeddings.to(_dev))
        _gpe = self._gpe.to(_dev) if self._gpe is not None else None
        pe_all  = (self.pe_proj(_gpe) if _gpe is not None
                   else torch.zeros(self.num_genes, DEPGPS_EMBED_DIM, device=_dev))

        _ei = self._ei.to(_dev)
        ea_dev = None
        if self.use_cdns and self.has_graph and self._ea is not None:
            ea_dev = self._ea.to(_dev)

        all_preds = []
        for b in range(len(cl_indices)):
            cl_idx = cl_indices[b]
            expr_tok = tokenize_expression(omics["expr"][cl_idx,:].cpu()).to(_dev)
            mut_tok  = tokenize_mut(omics["mut"][cl_idx,:].cpu()).to(_dev)
            cnv_tok  = tokenize_cnv(omics["cnv"][cl_idx,:].cpu()).to(_dev)

            feats = [self.expr_embed(expr_tok), self.mut_embed(mut_tok),
                     self.cnv_embed(cnv_tok), g2v_all, pe_all]
            if self.cdns_node_proj is not None:
                feats.append(self.cdns_node_proj(
                    cdns_gene_scalar.to(_dev)))
            x = torch.cat(feats, dim=1)

            for i in range(GNN_LAYERS):
                branch_out = None

                if self.has_graph:
                    x_gnn = self.gnn_layers[i](x, _ei)
                    if self.use_cdns and ea_dev is not None:
                        # scatter cdNS edge signal onto target nodes
                        tgt      = _ei[1]
                        edge_msg = self.edge_gates[i](ea_dev)  # [E, H]
                        agg      = torch.zeros(x.size(0), DEPGPS_GNN_HIDDEN, device=_dev)
                        agg.index_add_(0, tgt, edge_msg)
                        x_gnn = x_gnn + agg
                    x_gnn     = self.dropout(F.relu(self.gnn_norms[i](x_gnn)))
                    branch_out = x_gnn

                if self.use_perf:
                    x_perf = F.relu(
                        self.perf_norms[i](
                            self.perf_layers[i](
                                self.perf_in_proj[i](x).unsqueeze(0)).squeeze(0)))
                    branch_out = x_perf if branch_out is None else branch_out + x_perf

                res = self.residual_projs[i](x)
                x   = branch_out + res

            all_preds.append(self.output_head(x).squeeze(1))
        return torch.stack(all_preds, dim=0), None


# ---------------------------------------------------------------------------
# C. DeepDEP - with optional cdNS node-level feature
#
#    use_cdns=True appends a per-gene cdNS scalar to GeneIdentityEncoder input.
#    This is the only sensible analogue for a graph-free model.
# ---------------------------------------------------------------------------

class OmicsAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=DEEPDEP_AE_HIDDEN,
                 latent_dim=DEEPDEP_AE_LATENT, dropout=DEEPDEP_DROPOUT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
    def encode(self, x): return self.encoder(x)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class GeneIdentityEncoder(nn.Module):
    """
    Approximation of DeepDEP's MSigDB CGP fingerprint (3115-dim binary vector).
    We use gene2vec (200-dim) + Laplacian PE (16-dim) as a feasible substitute
    since MSigDB CGP fingerprint construction requires the MSigDB gene set
    database; this approximation is documented in our methods.
    use_cdns=True concatenates a z-normalised cdNS scalar ? 217-dim total.
    """
    def __init__(self, use_cdns=False):
        super().__init__()
        self.use_cdns = use_cdns
        input_dim = GENE2VEC_DIM + PE_DIM_TOTAL + (1 if use_cdns else 0)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, DEEPDEP_GENE_EMBED_DIM * 2), nn.ReLU(),
            nn.Dropout(DEEPDEP_DROPOUT),
            nn.Linear(DEEPDEP_GENE_EMBED_DIM * 2, DEEPDEP_GENE_EMBED_DIM))

    def forward(self, g2v, pe, cdns_scalar=None):
        feats = [g2v, pe]
        if self.use_cdns and cdns_scalar is not None:
            feats.append(cdns_scalar)
        return self.proj(torch.cat(feats, dim=1))


class DeepDEPModel(nn.Module):
    def __init__(self, num_genes, use_cdns=False):
        super().__init__()
        self.num_genes = num_genes
        self.use_cdns  = use_cdns
        self.ae_expr   = OmicsAutoencoder(num_genes)
        self.ae_mut    = OmicsAutoencoder(num_genes)
        self.ae_cnv    = OmicsAutoencoder(num_genes)
        self.gene_id   = GeneIdentityEncoder(use_cdns=use_cdns)
        concat_dim     = DEEPDEP_AE_LATENT * 3 + DEEPDEP_GENE_EMBED_DIM
        self.pred_mlp  = nn.Sequential(
            nn.Linear(concat_dim, DEEPDEP_MLP_HIDDEN), nn.ReLU(),
            nn.Dropout(DEEPDEP_DROPOUT),
            nn.Linear(DEEPDEP_MLP_HIDDEN, DEEPDEP_MLP_HIDDEN // 2), nn.ReLU(),
            nn.Dropout(DEEPDEP_DROPOUT),
            nn.Linear(DEEPDEP_MLP_HIDDEN // 2, 1))

    def set_gene_pe(self, gene_pe):
        self._gpe = gene_pe

    def forward(self, cl_indices, omics=None, edge_index=None, edge_attr=None,
                gene_pe=None):
        _dev = cl_indices.device
        if omics is None:
            omics = globals()['omics']
        _gpe = getattr(self, '_gpe', None)
        g2v  = gene2vec_embeddings.to(_dev)
        pe   = (_gpe.to(_dev) if _gpe is not None
                else torch.zeros(self.num_genes, PE_DIM_TOTAL, device=_dev))
        cs   = cdns_gene_scalar.to(_dev) if self.use_cdns else None
        gene_id_emb = self.gene_id(g2v, pe, cs)

        all_preds = []
        for b in range(len(cl_indices)):
            cl_idx = cl_indices[b]
            _, z_expr = self.ae_expr(omics["expr"][cl_idx,:].to(_dev).unsqueeze(0))
            _, z_mut  = self.ae_mut( omics["mut"][cl_idx,:].to(_dev).unsqueeze(0))
            _, z_cnv  = self.ae_cnv( omics["cnv"][cl_idx,:].to(_dev).unsqueeze(0))
            pair_in = torch.cat([
                z_expr.expand(self.num_genes, -1),
                z_mut.expand(self.num_genes,  -1),
                z_cnv.expand(self.num_genes,  -1),
                gene_id_emb], dim=1)
            all_preds.append(self.pred_mlp(pair_in).squeeze(1))
        return torch.stack(all_preds, dim=0), None


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def wrap_dp(model):
    if USE_DATA_PARALLEL:
        return nn.DataParallel(model, device_ids=GPU_IDS)
    return model

def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

log.info("Model architectures defined.")

# =============================================================================
# HYPERPARAMETER SWEEPS
# Each variant gets its own full lr x dropout grid (or lr-only for DeepDEP).
# Checkpointed so re-runs skip already-done sweeps.
#
# Every temp model is wrapped in wrap_dp(...) so hyperparameter sweeps use
# all available GPUs. Each iteration is wrapped in try/except
# torch.cuda.OutOfMemoryError so a single failed configuration is skipped
# and logged rather than terminating the full sweep.
# =============================================================================

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def quick_hp_eval(model, lr, ea_for_hp):
    """Train HP_EPOCHS epochs, return val Pearson."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(HP_EPOCHS):
        for cl_indices, scores, masks in hp_train_loader:
            cl_indices = cl_indices.to(device)
            scores     = scores.to(device)
            masks      = masks.to(device)
            preds, _   = model(cl_indices=cl_indices, edge_index=edge_index,
                               edge_attr=ea_for_hp, gene_pe=gene_pe)
            loss = F.mse_loss(preds[masks], scores[masks])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    vp_list, vl_list = [], []
    with torch.no_grad():
        for cl_indices, scores, masks in val_loader:
            cl_indices = cl_indices.to(device)
            scores     = scores.to(device)
            masks      = masks.to(device)
            preds, _   = model(cl_indices=cl_indices, edge_index=edge_index, edge_attr=ea_for_hp, gene_pe=gene_pe)
            vp_list.append(preds[masks].cpu().numpy())
            vl_list.append(scores[masks].cpu().numpy())
    vp = np.concatenate(vp_list); vl = np.concatenate(vl_list)
    return pearsonr(vp, vl)[0]


# --- GeneNetPred ---
ckpt_hp_gnp = load_checkpoint("hp_genenetpred")
if ckpt_hp_gnp is not None:
    LEARNING_RATE = ckpt_hp_gnp["best_lr"]
    DROPOUT       = ckpt_hp_gnp["best_dropout"]
    log.info(f"GeneNetPred HP loaded: lr={LEARNING_RATE} do={DROPOUT}")
else:
    log.info("GeneNetPred HP sweep...")
    best_lr_gnp, best_do_gnp, best_vp_gnp = LEARNING_RATE, DROPOUT, -1.0
    for lr in HP_LR_GRID:
        for do in HP_DROPOUT_GRID:
            DROPOUT = do
            try:
                m_tmp = GeneNetPred(num_genes, "sage", False, True); m_tmp.set_graph(edge_index, edge_attr); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
                vp    = quick_hp_eval(m_tmp, lr, edge_attr)
            except torch.cuda.OutOfMemoryError:
                log.info(f"  [GNP] lr={lr:.0e} do={do} -> OOM, skipping")
                del m_tmp; free_memory(); continue
            log.info(f"  [GNP] lr={lr:.0e} do={do} -> {vp:.4f}")
            if vp > best_vp_gnp: best_vp_gnp = vp; best_lr_gnp = lr; best_do_gnp = do
            del m_tmp; free_memory()
    DROPOUT = best_do_gnp; LEARNING_RATE = best_lr_gnp
    save_checkpoint("hp_genenetpred", {"best_lr": LEARNING_RATE, "best_dropout": DROPOUT,
                                       "best_pearson": best_vp_gnp})
    log.info(f"GeneNetPred best: lr={LEARNING_RATE} do={DROPOUT} r={best_vp_gnp:.4f}")

ckpt_hp_gnp_perf = load_checkpoint("hp_genenetpred_performer")
if ckpt_hp_gnp_perf is not None:
    LEARNING_RATE_PERF = ckpt_hp_gnp_perf["best_lr"]
    log.info(f"GeneNetPred+Perf HP loaded: lr={LEARNING_RATE_PERF}")
else:
    log.info("GeneNetPred+Performer HP sweep...")
    best_lr_perf, best_vp_perf = HP_LR_GRID_PERF[0], -1.0
    for lr in HP_LR_GRID_PERF:
        try:
            m_tmp = GeneNetPred(num_genes, "sage", True, True); m_tmp.set_graph(edge_index, edge_attr); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
            vp    = quick_hp_eval(m_tmp, lr, edge_attr)
        except torch.cuda.OutOfMemoryError:
            log.info(f"  [GNP+Perf] lr={lr:.0e} -> OOM, skipping")
            del m_tmp; free_memory(); continue
        log.info(f"  [GNP+Perf] lr={lr:.0e} -> {vp:.4f}")
        if vp > best_vp_perf: best_vp_perf = vp; best_lr_perf = lr
        del m_tmp; free_memory()
    LEARNING_RATE_PERF = best_lr_perf
    save_checkpoint("hp_genenetpred_performer",
                    {"best_lr": LEARNING_RATE_PERF, "best_pearson": best_vp_perf})
    log.info(f"GeneNetPred+Perf best: lr={LEARNING_RATE_PERF} r={best_vp_perf:.4f}")

# --- DepGPS: full lr x dropout grid for each of the 8 variants ---
depgps_best_hp = {}   # key: (internal_name, cdns) -> {lr, dropout}

for (vname, gnn_type, use_perf, _) in DEPGPS_VARIANTS:
    for use_cdns in [False, True]:
        hp_key = f"hp_depgps_{vname}_cdns{int(use_cdns)}"
        ckpt   = load_checkpoint(hp_key)
        run_key = (vname, use_cdns)
        if ckpt is not None:
            depgps_best_hp[run_key] = {"lr": ckpt["best_lr"],
                                        "dropout": ckpt["best_dropout"]}
            log.info(f"DepGPS HP loaded {hp_key}: lr={ckpt['best_lr']} do={ckpt['best_dropout']}")
            continue

        log.info(f"DepGPS HP sweep: {hp_key} ...")
        ea_hp = edge_attr if use_cdns else torch.zeros_like(edge_attr)
        best_lr_dg, best_do_dg, best_vp_dg = LEARNING_RATE, DEPGPS_DROPOUT, -1.0
        for lr in HP_LR_GRID:
            for do in HP_DROPOUT_GRID:
                DEPGPS_DROPOUT = do
                try:
                    m_tmp = DepGPSModel(num_genes, gnn_type=gnn_type,
                                        use_perf=use_perf, use_cdns=use_cdns)
                    m_tmp.set_graph(edge_index, ea_hp); m_tmp.set_gene_pe(gene_pe)
                    m_tmp = m_tmp.to(device)
                    vp    = quick_hp_eval(m_tmp, lr, ea_hp)
                except torch.cuda.OutOfMemoryError:
                    log.info(f"  lr={lr:.0e} do={do} -> OOM, skipping")
                    del m_tmp; free_memory(); continue
                log.info(f"  lr={lr:.0e} do={do} -> {vp:.4f}")
                if vp > best_vp_dg: best_vp_dg = vp; best_lr_dg = lr; best_do_dg = do
                del m_tmp; free_memory()
        depgps_best_hp[run_key] = {"lr": best_lr_dg, "dropout": best_do_dg}
        save_checkpoint(hp_key, {"best_lr": best_lr_dg, "best_dropout": best_do_dg,
                                  "best_pearson": best_vp_dg})
        log.info(f"  Best: lr={best_lr_dg} do={best_do_dg} r={best_vp_dg:.4f}")
        DEPGPS_DROPOUT = 0.3  # reset to default after each sweep

# --- DeepDEP: lr-only sweep for each of 2 variants ---
deepdep_best_hp = {}   # key: use_cdns -> {lr}

for (vname, use_cdns, _) in DEEPDEP_VARIANTS:
    hp_key = f"hp_deepdep_cdns{int(use_cdns)}"
    ckpt   = load_checkpoint(hp_key)
    if ckpt is not None:
        deepdep_best_hp[use_cdns] = {"lr": ckpt["best_lr"]}
        log.info(f"DeepDEP HP loaded {hp_key}: lr={ckpt['best_lr']}")
        continue
    log.info(f"DeepDEP HP sweep: {hp_key} ...")
    best_lr_dd, best_vp_dd = LEARNING_RATE, -1.0
    for lr in HP_LR_GRID:
        try:
            m_tmp = DeepDEPModel(num_genes, use_cdns=use_cdns); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
            vp    = quick_hp_eval(m_tmp, lr, None)
        except torch.cuda.OutOfMemoryError:
            log.info(f"  lr={lr:.0e} -> OOM, skipping")
            del m_tmp; free_memory(); continue
        log.info(f"  lr={lr:.0e} -> {vp:.4f}")
        if vp > best_vp_dd: best_vp_dd = vp; best_lr_dd = lr
        del m_tmp; free_memory()
    deepdep_best_hp[use_cdns] = {"lr": best_lr_dd}
    save_checkpoint(hp_key, {"best_lr": best_lr_dd, "best_pearson": best_vp_dd})
    log.info(f"  Best: lr={best_lr_dd} r={best_vp_dd:.4f}")

log.info("All HP sweeps complete.")

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, omics, edge_index, edge_attr, device,
              optimizer=None, gene_weights=None, gene_pe=None):
    training   = optimizer is not None
    core_model = unwrap(model)
    model.train() if training else model.eval()

    total_loss = 0.0; n_samples = 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for cl_indices, scores, masks in loader:
            cl_indices = cl_indices.to(device)
            scores     = scores.to(device)
            masks      = masks.to(device)

            preds, _ = model(cl_indices=cl_indices, edge_index=edge_index, edge_attr=edge_attr, gene_pe=gene_pe)
            vp = preds[masks]; vl = scores[masks]

            w_base = torch.where(vl < WEIGHT_THRESHOLD,
                                 torch.tensor(WEIGHT_VALUE, device=device),
                                 torch.tensor(1.0, device=device))
            if gene_weights is not None and training:
                gi_exp     = torch.arange(num_genes, device=device).unsqueeze(0).expand(
                    len(cl_indices), -1)
                flat_gi    = gi_exp[masks]
                w          = w_base * gene_weights.to(device)[flat_gi]
            else:
                w = w_base

            mse_l  = (F.mse_loss(vp, vl, reduction='none') * w).mean()
            pear_l = 1 - ((vp - vp.mean()) * (vl - vl.mean())).mean() / \
                         (vp.std() * vl.std() + 1e-6)
            loss   = mse_l + 0.1 * pear_l

            if training:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * vl.shape[0]
            n_samples  += vl.shape[0]
            all_preds.append(vp.detach().cpu().numpy())
            all_labels.append(vl.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    r, _       = pearsonr(all_preds, all_labels)
    return total_loss / n_samples, r


def train_model(model, model_name, train_loader, val_loader,
                omics, edge_index, edge_attr, device,
                use_cdns_weights=False, gene_pe=None, lr=None):
    ckpt = load_model_checkpoint(f"trained_{model_name}")
    if ckpt is not None and "meta" in ckpt:
        unwrap(model).load_state_dict(
            {k: v.to(device) for k, v in ckpt["state_dict"].items()}, strict=False)
        meta = ckpt["meta"]
        log.info(f"  {model_name} loaded: val_r={meta['val_pearson']:.4f}")
        return model, meta["train_pearson"], meta["train_mse"], \
               meta["val_pearson"], meta["val_mse"]

    use_lr    = lr if lr is not None else LEARNING_RATE
    core_mdl  = unwrap(model)
    optimizer = torch.optim.Adam(core_mdl.parameters(), lr=use_lr,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_warmup_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    gw        = cdns_gene_weights if use_cdns_weights else None

    best_val_r   = -1.0; best_val_mse = float('inf')
    best_train_r = -1.0; best_train_mse = float('inf')
    patience_ctr = 0
    best_state   = {k: v.clone() for k, v in core_mdl.state_dict().items()}
    train_curve, val_curve = [], []

    log.info(f"\nTraining {model_name}  lr={use_lr}  cdNS_wts={use_cdns_weights}")
    log.info(f"{'Ep':>4} {'Tr MSE':>8} {'Tr r':>7} {'Va MSE':>8} {'Va r':>7}")
    log.info("-" * 42)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_mse, tr_r = run_epoch(model, train_loader, omics, edge_index, edge_attr,
                                  device, optimizer, gw, gene_pe)
        va_mse, va_r = run_epoch(model, val_loader,   omics, edge_index, edge_attr,
                                  device, gene_pe=gene_pe)
        scheduler.step()
        train_curve.append(tr_r); val_curve.append(va_r)
        flag = " *" if va_r > best_val_r else ""
        log.info(f"{epoch:>4} {tr_mse:>8.4f} {tr_r:>7.4f} {va_mse:>8.4f} {va_r:>7.4f}{flag}")

        if va_r > best_val_r:
            best_val_r   = va_r; best_val_mse   = va_mse
            best_train_r = tr_r; best_train_mse = tr_mse
            best_state   = {k: v.clone() for k, v in core_mdl.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"  Early stopping at epoch {epoch}"); break

    core_mdl.load_state_dict(best_state)
    meta = {"val_pearson": best_val_r, "val_mse": best_val_mse,
            "train_pearson": best_train_r, "train_mse": best_train_mse}
    save_model_checkpoint(f"trained_{model_name}", best_state, meta)

    plt.figure(figsize=(8, 4))
    plt.plot(train_curve, label='Train'); plt.plot(val_curve, label='Val')
    plt.xlabel("Epoch"); plt.ylabel("Pearson")
    plt.title(f"{model_name} Learning Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/curve_{model_name}.png", dpi=150); plt.close()

    log.info(f"  Best Train={best_train_r:.4f}  Val={best_val_r:.4f}")
    return model, best_train_r, best_train_mse, best_val_r, best_val_mse


def train_teacher_student(teacher, student, model_name,
                          train_loader, val_loader,
                          omics, edge_index, edge_attr, device,
                          distill_alpha=0.3, gene_pe=None):
    ckpt = load_model_checkpoint(f"trained_{model_name}")
    if ckpt is not None and "meta" in ckpt:
        unwrap(student).load_state_dict(
            {k: v.to(device) for k, v in ckpt["state_dict"].items()}, strict=False)
        log.info(f"  {model_name} loaded: val_r={ckpt['meta']['val_pearson']:.4f}")
        return student, ckpt["meta"]["val_pearson"]

    for p in unwrap(teacher).parameters(): p.requires_grad = False
    teacher.eval()

    core_student = unwrap(student)
    optimizer    = torch.optim.Adam(core_student.parameters(),
                                    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler    = get_cosine_warmup_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    plain_ea     = torch.zeros_like(edge_attr)
    gw           = cdns_gene_weights

    best_val_r = -1.0; patience_ctr = 0
    best_state = {k: v.clone() for k, v in core_student.state_dict().items()}

    log.info(f"\nTeacher-Student: {model_name}")
    for epoch in range(1, NUM_EPOCHS + 1):
        student.train(); ep_task = ep_dist = n = 0
        for cl_indices, scores, masks in train_loader:
            cl_indices = cl_indices.to(device)
            scores     = scores.to(device)
            masks      = masks.to(device)
            s_preds, _ = student(cl_indices=cl_indices, edge_index=edge_index, edge_attr=edge_attr, gene_pe=gene_pe)
            with torch.no_grad():
                t_preds, _ = teacher(cl_indices=cl_indices, edge_index=edge_index, edge_attr=plain_ea, gene_pe=gene_pe)
            vp = s_preds[masks]; vl = scores[masks]; tp = t_preds[masks]
            w_base    = torch.where(vl < WEIGHT_THRESHOLD,
                                    torch.tensor(WEIGHT_VALUE, device=device),
                                    torch.tensor(1.0, device=device))
            gi_exp    = torch.arange(num_genes, device=device).unsqueeze(0).expand(
                len(cl_indices), -1)
            w         = w_base * gw.to(device)[gi_exp[masks]]
            task_l    = (F.mse_loss(vp, vl, reduction='none') * w).mean() + \
                        0.1 * (1 - ((vp - vp.mean()) * (vl - vl.mean())).mean() /
                               (vp.std() * vl.std() + 1e-6))
            distil_l  = F.mse_loss(vp, tp.detach())
            loss      = (1 - distill_alpha) * task_l + distill_alpha * distil_l
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(core_student.parameters(), 1.0)
            optimizer.step()
            ep_task += task_l.item() * vl.shape[0]
            ep_dist += distil_l.item() * vl.shape[0]; n += vl.shape[0]

        scheduler.step()
        _, va_r = run_epoch(student, val_loader, omics, edge_index,
                             edge_attr, device, gene_pe=gene_pe)
        flag = " *" if va_r > best_val_r else ""
        log.info(f"  Ep {epoch:>3}  task={ep_task/n:.4f}  distil={ep_dist/n:.4f}"
                 f"  val_r={va_r:.4f}{flag}")
        if va_r > best_val_r:
            best_val_r = va_r
            best_state = {k: v.clone() for k, v in core_student.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE: log.info("  Early stopping."); break

    core_student.load_state_dict(best_state)
    save_model_checkpoint(f"trained_{model_name}", best_state,
                          {"val_pearson": best_val_r})
    return student, best_val_r

# =============================================================================
# TRAIN ALL MODELS
# =============================================================================

log.info("\n" + "=" * 65)
log.info("Training all models...")
log.info("=" * 65)
training_start = time.time()

results_val_pearson   = {}
results_train_pearson = {}

# --- GeneNetPred ---
model_sage      = GeneNetPred(num_genes, "sage", False, True); model_sage.set_graph(edge_index, edge_attr); model_sage.set_gene_pe(gene_pe); model_sage = wrap_dp(model_sage.to(device))
model_gcn       = GeneNetPred(num_genes, "gcn",  False, True); model_gcn.set_graph(edge_index, edge_attr); model_gcn.set_gene_pe(gene_pe); model_gcn = wrap_dp(model_gcn.to(device))
model_mlp       = wrap_dp(MLPBaseline().to(device))
model_sage_perf = GeneNetPred(num_genes, "sage", True,  True); model_sage_perf.set_graph(edge_index, edge_attr); model_sage_perf.set_gene_pe(gene_pe); model_sage_perf = wrap_dp(model_sage_perf.to(device))
model_gcn_perf  = GeneNetPred(num_genes, "gcn",  True,  True); model_gcn_perf.set_graph(edge_index, edge_attr); model_gcn_perf.set_gene_pe(gene_pe); model_gcn_perf = wrap_dp(model_gcn_perf.to(device))

for mdl, name, lr, use_cdns in [
    (model_sage,      "GNP_SAGE",          LEARNING_RATE,      True),
    (model_gcn,       "GNP_GCN",           LEARNING_RATE,      True),
    (model_mlp,       "MLP_Baseline",       LEARNING_RATE,      False),
    (model_sage_perf, "GNP_SAGE_Performer", LEARNING_RATE_PERF, True),
    (model_gcn_perf,  "GNP_GCN_Performer",  LEARNING_RATE_PERF, True),
]:
    mdl, tr_r, tr_m, va_r, va_m = train_model(
        mdl, name, train_loader, val_loader,
        omics, edge_index, edge_attr, device,
use_cdns_weights=use_cdns, gene_pe=gene_pe, lr=lr)
results_train_pearson[name] = tr_r
results_val_pearson[name]   = va_r
free_memory()

# Teacher-Student
teacher_model = GeneNetPred(num_genes, "sage", True, False); teacher_model.set_graph(edge_index, torch.zeros_like(edge_attr)); teacher_model.set_gene_pe(gene_pe); teacher_model = wrap_dp(teacher_model.to(device))
teacher_model, *_ = train_model(
teacher_model, "Teacher_SAGE_Performer",
train_loader, val_loader, omics, edge_index,
torch.zeros_like(edge_attr), device,
use_cdns_weights=False, gene_pe=gene_pe, lr=LEARNING_RATE_PERF)
del teacher_model; free_memory()
free_memory()



# --- DepGPS: all 8 variants ---
log.info("\n--- DepGPS: all 4 architectures x cdNS on/off ---")
depgps_trained = {}   # key: (vname, use_cdns) -> trained model

for (vname, gnn_type, use_perf, display_label) in DEPGPS_VARIANTS:
    for use_cdns in [False, True]:
        run_key   = (vname, use_cdns)
        hp        = depgps_best_hp[run_key]
        train_ea  = edge_attr if use_cdns else torch.zeros_like(edge_attr)
        mdl_name  = f"{vname}_cdns{int(use_cdns)}"
        cdns_tag  = "+cdNS" if use_cdns else "noCDNS"
        full_label = f"{display_label} [{cdns_tag}]"

        m = DepGPSModel(num_genes, gnn_type=gnn_type, use_perf=use_perf, use_cdns=use_cdns); m.set_graph(edge_index, train_ea); m.set_gene_pe(gene_pe); m = wrap_dp(m.to(device))
        m, tr_r, tr_m, va_r, va_m = train_model(
            m, mdl_name, train_loader, val_loader,
            omics, edge_index, train_ea, device,
            use_cdns_weights=False, gene_pe=gene_pe, lr=hp["lr"])
        depgps_trained[run_key]             = (m, train_ea, full_label)
        results_val_pearson[full_label]     = va_r
        results_train_pearson[full_label]   = tr_r
        free_memory()

# --- DeepDEP: 2 variants ---
log.info("\n--- DeepDEP: no cdNS vs +cdNS ---")
deepdep_trained = {}   # key: use_cdns -> trained model

for (vname, use_cdns, display_label) in DEEPDEP_VARIANTS:
    hp  = deepdep_best_hp[use_cdns]
    m = DeepDEPModel(num_genes, use_cdns=use_cdns); m.set_gene_pe(gene_pe); m = wrap_dp(m.to(device))
    m, tr_r, tr_m, va_r, va_m = train_model(
        m, vname, train_loader, val_loader,
        omics, edge_index, None, device,
        use_cdns_weights=False, gene_pe=gene_pe, lr=hp["lr"])
    deepdep_trained[use_cdns]              = (m, display_label)
    results_val_pearson[display_label]     = va_r
    results_train_pearson[display_label]   = tr_r
    free_memory()

total_training_time = time.time() - training_start
log.info(f"\nTotal training time: {total_training_time/3600:.2f} hours")

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, loader, omics, edge_index, edge_attr, device,
                    gene_pe=None):
    model.eval()
    all_preds, all_labels, all_cl, all_g, all_attn = [], [], [], [], []
    with torch.no_grad():
        for cl_indices, scores, masks in loader:
            cl_indices = cl_indices.to(device)
            scores     = scores.to(device)
            masks      = masks.to(device)
            preds, attns = model(cl_indices=cl_indices, edge_index=edge_index, edge_attr=edge_attr, gene_pe=gene_pe)
            for b in range(len(cl_indices)):
                m = masks[b]
                all_preds.append(preds[b][m].cpu().numpy())
                all_labels.append(scores[b][m].cpu().numpy())
                all_cl.append(np.full(m.sum().item(), cl_indices[b].item()))
                all_g.append(torch.where(m)[0].cpu().numpy())
                if attns is not None:
                    all_attn.append(attns[b][m].cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels),
            np.concatenate(all_cl),   np.concatenate(all_g),
            np.concatenate(all_attn) if all_attn else None)


def calculate_metrics(preds, labels):
    p,  _ = pearsonr(preds, labels)
    sp, _ = spearmanr(preds, labels)
    mse   = np.mean((preds - labels) ** 2)
    return p, sp, mse, np.sqrt(mse)


# Build the evaluation registry
eval_models = {}
eval_models["GeneNetPred (SAGE+Performer)"] = (model_sage_perf, edge_attr)
eval_models["GeneNetPred (GCN+Performer)"]  = (model_gcn_perf,  edge_attr)
eval_models["GeneNetPred (SAGE)"]            = (model_sage,      edge_attr)
eval_models["GeneNetPred (GCN)"]             = (model_gcn,       edge_attr)
eval_models["MLP Baseline"]                  = (model_mlp,       edge_attr)

for run_key, (m, ea, label) in depgps_trained.items():
    eval_models[label] = (m, ea)

for use_cdns, (m, label) in deepdep_trained.items():
    eval_models[label] = (m, None)

all_test_results = {}
log.info("\n" + "=" * 70)
log.info("Evaluating all models on test set...")
for name, (mdl, ea) in eval_models.items():
    tp, tl, tc, tg, ta = evaluate_model(
        mdl, test_loader, omics, edge_index, ea, device, gene_pe=gene_pe)
    p, sp, m, rm = calculate_metrics(tp, tl)
    all_test_results[name] = {
        "pearson": p, "spearman": sp, "mse": m, "rmse": rm,
        "preds": tp, "labels": tl, "cl_idx": tc, "g_idx": tg}
    log.info(f"  {name:<50} r={p:.4f}  sr={sp:.4f}  MSE={m:.4f}")

# Best single GNP model for downstream figures
best_gnp_name = max(
    [k for k in all_test_results if "GeneNetPred" in k],
    key=lambda x: all_test_results[x]["pearson"])
fig_model, fig_ea = eval_models[best_gnp_name]
fig_data          = all_test_results[best_gnp_name]
preds, labels     = fig_data["preds"], fig_data["labels"]
g_idx, cl_idx_arr = fig_data["g_idx"], fig_data["cl_idx"]
pearson           = fig_data["pearson"]
tp2, tl2, tc2, tg2, ta2 = evaluate_model(
    fig_model, test_loader, omics, edge_index, fig_ea, device, gene_pe=gene_pe)
attn_weights_arr = ta2

# =============================================================================
# COMPARISON TABLE  -  model x cdNS(on/off) axis
# =============================================================================

log.info("\n" + "=" * 100)
log.info("  FAIR COMPARISON TABLE - identical dataset & split for all models")
log.info("=" * 100)
log.info(f"  {'Model':<55} {'cdNS':>6} {'Pearson':>8} {'Spearman':>10} "
         f"{'MSE':>8} {'RMSE':>8}")
log.info("-" * 100)

def print_row(name, cdns_flag, r):
    flag = all_test_results.get(name, {})
    p  = r.get("pearson",  float("nan"))
    sp = r.get("spearman", float("nan"))
    m  = r.get("mse",      float("nan"))
    rm = r.get("rmse",     float("nan"))
    log.info(f"  {name:<55} {cdns_flag:>6} {p:>8.4f} {sp:>10.4f} {m:>8.4f} {rm:>8.4f}")

log.info("  >> GeneNetPred")
for n in ["GeneNetPred (SAGE+Performer)", "GeneNetPred (GCN+Performer)",
          "GeneNetPred (SAGE)", "GeneNetPred (GCN)", "MLP Baseline"]:
    if n in all_test_results:
        cdns = "yes" if n != "MLP Baseline" else "n/a"
        print_row(n, cdns, all_test_results[n])

log.info("-" * 100)
log.info("  >> DepGPS (all 4 paper variants x cdNS on/off)")
for (vname, gnn_type, use_perf, display_label) in DEPGPS_VARIANTS:
    for use_cdns in [False, True]:
        cdns_tag   = "+cdNS" if use_cdns else "noCDNS"
        full_label = f"{display_label} [{cdns_tag}]"
        cdns_col   = "yes" if use_cdns else "no"
        if full_label in all_test_results:
            print_row(full_label, cdns_col, all_test_results[full_label])

log.info("-" * 100)
log.info("  >> DeepDEP (cdNS injected as gene-level node feature)")
for (vname, use_cdns, display_label) in DEEPDEP_VARIANTS:
    cdns_col = "yes" if use_cdns else "no"
    if display_label in all_test_results:
        print_row(display_label, cdns_col, all_test_results[display_label])

log.info("=" * 100)

# Save to CSV
comparison_rows = []
for name, r in all_test_results.items():
    comparison_rows.append({
        "model":    name,
        "pearson":  round(r.get("pearson",  float("nan")), 4),
        "spearman": round(r.get("spearman", float("nan")), 4),
        "mse":      round(r.get("mse",      float("nan")), 4),
        "rmse":     round(r.get("rmse",     float("nan")), 4),
        "dataset":  "ours (same split)",
    })
pd.DataFrame(comparison_rows).to_csv(
    f"{OUTPUT_DIR}/comparison_table_v8.csv", index=False)
log.info(f"  Comparison table saved.")

# Grouped bar chart: DepGPS cdNS on vs off for each variant
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: DepGPS 4-variant x cdNS
dg_labels, dg_no_cdns, dg_with_cdns = [], [], []
for (vname, gnn_type, use_perf, display_label) in DEPGPS_VARIANTS:
    dg_labels.append(display_label)
    label_no   = f"{display_label} [noCDNS]"
    label_yes  = f"{display_label} [+cdNS]"
    dg_no_cdns.append(all_test_results.get(label_no,  {}).get("pearson", 0))
    dg_with_cdns.append(all_test_results.get(label_yes, {}).get("pearson", 0))

x     = np.arange(len(dg_labels))
width = 0.35
axes[0].bar(x - width/2, dg_no_cdns,   width, label='no cdNS', color='steelblue',  alpha=0.85)
axes[0].bar(x + width/2, dg_with_cdns, width, label='+cdNS',   color='darkorange', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(dg_labels, rotation=20, ha='right')
axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("Test Pearson")
axes[0].set_title("DepGPS - All Variants x cdNS (same dataset/split)")
axes[0].legend()
for i, (v1, v2) in enumerate(zip(dg_no_cdns, dg_with_cdns)):
    axes[0].text(i - width/2, v1 + 0.01, f"{v1:.3f}", ha='center', fontsize=7)
    axes[0].text(i + width/2, v2 + 0.01, f"{v2:.3f}", ha='center', fontsize=7)

# Right: DeepDEP + best GNP
right_names  = ["DeepDEP (no cdNS)", "DeepDEP (+cdNS)", best_gnp_name]
right_vals   = [all_test_results.get(n, {}).get("pearson", 0) for n in right_names]
right_colors = ["seagreen", "mediumseagreen", "steelblue"]
bars = axes[1].bar(right_names, right_vals, color=right_colors, alpha=0.85)
axes[1].set_ylim(0, 1.0); axes[1].set_ylabel("Test Pearson")
axes[1].set_title("DeepDEP cdNS effect + Best GNP (same dataset/split)")
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, right_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_comparison_v8.png", dpi=150); plt.close()
log.info("Comparison figure saved.")

# =============================================================================
# ABLATION STUDY (GeneNetPred best variant)
# =============================================================================

log.info("\n" + "=" * 60)
log.info("Ablation Study...")

best_gnn_type      = "sage"
best_use_performer = True
if "GCN+Performer"  in best_gnp_name: best_gnn_type = "gcn"
elif "GCN" in best_gnp_name and "Performer" not in best_gnp_name:
    best_gnn_type = "gcn"; best_use_performer = False
elif "SAGE" in best_gnp_name and "Performer" not in best_gnp_name:
    best_use_performer = False


def train_ablation_quick(model, use_omics=None, use_edge_attr=None,
                          zero_gene2vec=False, use_gene_pe=None):
    if use_omics     is None: use_omics     = omics
    if use_edge_attr is None: use_edge_attr = edge_attr
    if use_gene_pe   is None: use_gene_pe   = gene_pe
    global gene2vec_embeddings
    orig_g2v = gene2vec_embeddings
    if zero_gene2vec:
        gene2vec_embeddings = torch.zeros_like(orig_g2v)
    opt        = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    best_vp    = -1.0; pc = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(1, ABLATION_EPOCHS + 1):
        run_epoch(model, train_loader, use_omics, edge_index,
                  use_edge_attr, device, opt, gene_pe=use_gene_pe)
        _, vp = run_epoch(model, val_loader, use_omics, edge_index,
                          use_edge_attr, device, gene_pe=use_gene_pe)
        if vp > best_vp: best_vp = vp; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pc = 0
        else:
            pc += 1
            if pc >= PATIENCE: break
    model.load_state_dict(best_state)
    tp, tl, _, _, _ = evaluate_model(model, test_loader, use_omics, edge_index,
                                      use_edge_attr, device, gene_pe=use_gene_pe)
    p, _, m, _ = calculate_metrics(tp, tl)
    if zero_gene2vec: gene2vec_embeddings = orig_g2v
    return p, m


def make_ablation_omics(remove_modality):
    a = {k: v.clone() for k, v in omics.items()}
    a[remove_modality] = torch.zeros_like(omics[remove_modality])
    return a


ablation_results = {}
ablation_results[f"Full ({best_gnp_name})"] = \
    (pearson, all_test_results[best_gnp_name]["mse"])

m_tmp = GeneNetPred(num_genes, best_gnn_type, best_use_performer, False); m_tmp.set_graph(edge_index, torch.zeros_like(edge_attr)); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
p, mv = train_ablation_quick(m_tmp, use_edge_attr=torch.zeros_like(edge_attr))
ablation_results["Without cdNS Edges"] = (p, mv); del m_tmp; free_memory()

m_tmp = GeneNetPred(num_genes, best_gnn_type, False, True); m_tmp.set_graph(edge_index, edge_attr); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
p, mv = train_ablation_quick(m_tmp)
ablation_results["Without Performer"] = (p, mv); del m_tmp; free_memory()

m_tmp = GeneNetPredNoAttention(num_genes, best_gnn_type, best_use_performer, True); m_tmp.set_graph(edge_index, edge_attr); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
p, mv = train_ablation_quick(m_tmp)
ablation_results["Without Attention"] = (p, mv); del m_tmp; free_memory()

tp_mlp, tl_mlp, _, _, _ = evaluate_model(
    model_mlp, test_loader, omics, edge_index, edge_attr, device, gene_pe=gene_pe)
p_mlp, _, mv_mlp, _ = calculate_metrics(tp_mlp, tl_mlp)
ablation_results["Without GNN (MLP)"] = (p_mlp, mv_mlp)

for mod in ["expr", "mut", "cnv"]:
    m_tmp = GeneNetPred(num_genes, best_gnn_type, best_use_performer, True); m_tmp.set_graph(edge_index, edge_attr); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
    p, mv = train_ablation_quick(m_tmp, use_omics=make_ablation_omics(mod))
    ablation_results[f"Without {mod.upper()}"] = (p, mv)
    del m_tmp; free_memory()

m_tmp = GeneNetPred(num_genes, best_gnn_type, best_use_performer, True); m_tmp.set_graph(edge_index, edge_attr); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
p, mv = train_ablation_quick(m_tmp, zero_gene2vec=True)
ablation_results["Without gene2vec"] = (p, mv); del m_tmp; free_memory()

m_tmp = GeneNetPred(num_genes, best_gnn_type, best_use_performer, True); m_tmp.set_graph(edge_index, edge_attr); m_tmp.set_gene_pe(gene_pe); m_tmp = m_tmp.to(device)
p, mv = train_ablation_quick(m_tmp, use_gene_pe=torch.zeros_like(gene_pe))
ablation_results["Without Positional Emb."] = (p, mv); del m_tmp; free_memory()

log.info("Ablation Results:")
for name, (p, m) in ablation_results.items():
    log.info(f"  {name:<38} r={p:.4f}  MSE={m:.4f}")

abl_names    = list(ablation_results.keys())
abl_pearsons = [ablation_results[n][0] for n in abl_names]
abl_colors   = ['steelblue' if 'Full' in n else 'lightcoral' for n in abl_names]
plt.figure(figsize=(17, 5))
bars = plt.bar(abl_names, abl_pearsons, color=abl_colors)
plt.ylabel("Test Pearson"); plt.title("Ablation Study (v8)")
plt.xticks(rotation=25, ha='right'); plt.ylim(0, 1.0)
for bar, val in zip(bars, abl_pearsons):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_ablation.png", dpi=150); plt.close()
log.info("Ablation figure saved.")

# =============================================================================
# MULTI-SEED EVALUATION  -  all DepGPS variants + DeepDEP variants, 10 seeds
# =============================================================================

log.info("\n" + "=" * 65)
log.info(f"Multi-Seed Evaluation ({len(SEEDS)} seeds) - all variants...")

# Build seed_results keys to match all 8 DepGPS + 2 DeepDEP + GNP variants
_gnp_seed_keys  = ["GNP_SAGE", "GNP_GCN", "GNP_SAGE+Perf", "GNP_GCN+Perf",
                    "Student (T-S)"]
_dg_seed_keys   = [f"{vname}_cdns{int(c)}"
                   for (vname, _, _, _) in DEPGPS_VARIANTS
                   for c in [False, True]]
_dd_seed_keys   = ["DeepDEP_noCDNS", "DeepDEP_cdNS"]
_all_seed_keys  = _gnp_seed_keys + _dg_seed_keys + _dd_seed_keys

ckpt_ms = load_checkpoint("multiseed_results_v8")
if ckpt_ms is not None:
    seed_results = ckpt_ms
    log.info("  Multi-seed results loaded from checkpoint.")
else:
    seed_results = {k: [] for k in _all_seed_keys}

    for seed in SEEDS:
        log.info(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed); np.random.seed(seed)

        # GeneNetPred variants
        for vname, gnn_type, use_perf, use_edge, use_lr in [
            ("GNP_SAGE",      "sage", False, True, LEARNING_RATE),
            ("GNP_GCN",       "gcn",  False, True, LEARNING_RATE),
            ("GNP_SAGE+Perf", "sage", True,  True, LEARNING_RATE_PERF),
            ("GNP_GCN+Perf",  "gcn",  True,  True, LEARNING_RATE_PERF),
        ]:
            m = GeneNetPred(num_genes, gnn_type, use_perf, use_edge); m.set_graph(edge_index, edge_attr); m.set_gene_pe(gene_pe); m = wrap_dp(m.to(device))
            opt = torch.optim.Adam(unwrap(m).parameters(), lr=use_lr,
                                   weight_decay=WEIGHT_DECAY)
            sch = get_cosine_warmup_scheduler(opt, WARMUP_EPOCHS, MULTISEED_EPOCHS)
            bvp = -1.0; pc = 0
            bst = {k: v.clone() for k, v in unwrap(m).state_dict().items()}
            for epoch in range(1, MULTISEED_EPOCHS + 1):
                run_epoch(m, train_loader, omics, edge_index, edge_attr,
                          device, opt, cdns_gene_weights, gene_pe)
                _, vp = run_epoch(m, val_loader, omics, edge_index, edge_attr,
                                  device, gene_pe=gene_pe)
                sch.step()
                if vp > bvp:
                    bvp = vp; bst = {k: v.clone() for k, v in unwrap(m).state_dict().items()}; pc = 0
                else:
                    pc += 1
                    if pc >= PATIENCE: break
            unwrap(m).load_state_dict(bst)
            tp, tl, _, _, _ = evaluate_model(m, test_loader, omics, edge_index,
                                              edge_attr, device, gene_pe=gene_pe)
            p, _, _, _ = calculate_metrics(tp, tl)
            seed_results[vname].append(p)
            log.info(f"  {vname} seed={seed}  r={p:.4f}")
            del m; free_memory()

        # Teacher-Student
        ts_t = GeneNetPred(num_genes, "sage", True, False); ts_t.set_graph(edge_index, torch.zeros_like(edge_attr)); ts_t.set_gene_pe(gene_pe); ts_t = wrap_dp(ts_t.to(device))
        opt  = torch.optim.Adam(unwrap(ts_t).parameters(), lr=LEARNING_RATE_PERF,
                                weight_decay=WEIGHT_DECAY)
        sch  = get_cosine_warmup_scheduler(opt, WARMUP_EPOCHS, MULTISEED_EPOCHS)
        bvp  = -1.0; pc = 0
        bst  = {k: v.clone() for k, v in unwrap(ts_t).state_dict().items()}
        for epoch in range(1, MULTISEED_EPOCHS + 1):
            run_epoch(ts_t, train_loader, omics, edge_index,
                      torch.zeros_like(edge_attr), device, opt, None, gene_pe)
            _, vp = run_epoch(ts_t, val_loader, omics, edge_index,
                              torch.zeros_like(edge_attr), device, gene_pe=gene_pe)
            sch.step()
            if vp > bvp:
                bvp = vp; bst = {k: v.clone() for k, v in unwrap(ts_t).state_dict().items()}; pc = 0
            else:
                pc += 1
                if pc >= PATIENCE: break
        unwrap(ts_t).load_state_dict(bst)
        for _p in unwrap(ts_t).parameters(): _p.requires_grad = False
        ts_t.eval()

        ts_s = GeneNetPred(num_genes, "sage", True, True); ts_s.set_graph(edge_index, edge_attr); ts_s.set_gene_pe(gene_pe); ts_s = wrap_dp(ts_s.to(device))
        ts_s, _ = train_teacher_student(
            ts_t, ts_s, f"Student_seed{seed}",
            train_loader, val_loader, omics, edge_index, edge_attr,
            device, gene_pe=gene_pe)
        tp, tl, _, _, _ = evaluate_model(ts_s, test_loader, omics, edge_index,
                                          edge_attr, device, gene_pe=gene_pe)
        p, _, _, _ = calculate_metrics(tp, tl)
        seed_results["Student (T-S)"].append(p)
        log.info(f"  Student (T-S) seed={seed}  r={p:.4f}")
        del ts_t, ts_s; free_memory()

        # DepGPS: all 8 variants
        for (vname, gnn_type, use_perf, _) in DEPGPS_VARIANTS:
            for use_cdns in [False, True]:
                run_key   = (vname, use_cdns)
                seed_key  = f"{vname}_cdns{int(use_cdns)}"
                hp        = depgps_best_hp[run_key]
                train_ea  = edge_attr if use_cdns else torch.zeros_like(edge_attr)
                m = DepGPSModel(num_genes, gnn_type=gnn_type,
                                          use_perf=use_perf, use_cdns=use_cdns)
                m.set_graph(edge_index, train_ea); m.set_gene_pe(gene_pe)
                m = wrap_dp(m.to(device))
                opt = torch.optim.Adam(unwrap(m).parameters(), lr=hp["lr"],
                                       weight_decay=WEIGHT_DECAY)
                sch = get_cosine_warmup_scheduler(opt, WARMUP_EPOCHS, MULTISEED_EPOCHS)
                bvp = -1.0; pc = 0
                bst = {k: v.clone() for k, v in unwrap(m).state_dict().items()}
                for epoch in range(1, MULTISEED_EPOCHS + 1):
                    run_epoch(m, train_loader, omics, edge_index,
                              train_ea, device, opt, None, gene_pe)
                    _, vp = run_epoch(m, val_loader, omics, edge_index,
                                      train_ea, device, gene_pe=gene_pe)
                    sch.step()
                    if vp > bvp:
                        bvp = vp; bst = {k: v.clone() for k, v in unwrap(m).state_dict().items()}; pc = 0
                    else:
                        pc += 1
                        if pc >= PATIENCE: break
                unwrap(m).load_state_dict(bst)
                tp, tl, _, _, _ = evaluate_model(m, test_loader, omics, edge_index,
                                                  train_ea, device, gene_pe=gene_pe)
                p, _, _, _ = calculate_metrics(tp, tl)
                seed_results[seed_key].append(p)
                log.info(f"  {seed_key} seed={seed}  r={p:.4f}")
                del m; free_memory()

        # DeepDEP: both variants
        for (vname, use_cdns, _) in DEEPDEP_VARIANTS:
            hp  = deepdep_best_hp[use_cdns]
            m = DeepDEPModel(num_genes, use_cdns=use_cdns); m.set_gene_pe(gene_pe); m = wrap_dp(m.to(device))
            opt = torch.optim.Adam(unwrap(m).parameters(), lr=hp["lr"],
                                   weight_decay=WEIGHT_DECAY)
            sch = get_cosine_warmup_scheduler(opt, WARMUP_EPOCHS, MULTISEED_EPOCHS)
            bvp = -1.0; pc = 0
            bst = {k: v.clone() for k, v in unwrap(m).state_dict().items()}
            for epoch in range(1, MULTISEED_EPOCHS + 1):
                run_epoch(m, train_loader, omics, edge_index,
                          None, device, opt, None, gene_pe)
                _, vp = run_epoch(m, val_loader, omics, edge_index,
                                  None, device, gene_pe=gene_pe)
                sch.step()
                if vp > bvp:
                    bvp = vp; bst = {k: v.clone() for k, v in unwrap(m).state_dict().items()}; pc = 0
                else:
                    pc += 1
                    if pc >= PATIENCE: break
            unwrap(m).load_state_dict(bst)
            tp, tl, _, _, _ = evaluate_model(m, test_loader, omics, edge_index,
                                              None, device, gene_pe=gene_pe)
            p, _, _, _ = calculate_metrics(tp, tl)
            seed_results[vname].append(p)
            log.info(f"  {vname} seed={seed}  r={p:.4f}")
            del m; free_memory()

        # Checkpoint after each seed so resume is safe
        save_checkpoint("multiseed_results_v8",
                        {k: [float(x) for x in v] for k, v in seed_results.items()})

log.info("\nMulti-Seed Summary (mean +/- std):")
for name, ps in seed_results.items():
    if ps:
        log.info(f"  {name:<35}  {np.mean(ps):.4f} +/- {np.std(ps):.4f}")

# Multi-seed figure: grouped by model family, colour by cdNS
fig, ax = plt.subplots(figsize=(22, 6))
_names = list(seed_results.keys())
_means = [np.mean(seed_results[n]) if seed_results[n] else 0 for n in _names]
_stds  = [np.std(seed_results[n])  if seed_results[n] else 0 for n in _names]
_colors = []
for n in _names:
    if "GNP" in n or "Student" in n:      _colors.append("steelblue")
    elif "cdns0" in n or "noCDNS" in n:   _colors.append("coral")
    else:                                   _colors.append("darkorange")
ax.bar(_names, _means, yerr=_stds, capsize=5, color=_colors, alpha=0.85)
ax.set_xticks(range(len(_names)))
ax.set_xticklabels(_names, rotation=30, ha='right', fontsize=7)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Test Pearson (mean +/- std)")
ax.set_title(f"Multi-Seed Evaluation ({len(SEEDS)} seeds) - All Variants (same dataset/split)")
for i, (m, s) in enumerate(zip(_means, _stds)):
    if m > 0:
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha='center', fontsize=6)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='steelblue',  label='GeneNetPred'),
    Patch(facecolor='coral',      label='Baseline (no cdNS)'),
    Patch(facecolor='darkorange', label='Baseline (+cdNS)'),
], fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_multiseed_v8.png", dpi=150); plt.close()
log.info("Multi-seed figure saved.")

# =============================================================================
# CANCER TYPE ANALYSIS
# =============================================================================

log.info("\n" + "=" * 60)
log.info("Cancer Type Analysis...")

test_cell_lines  = [cell_lines[i] for i in test_idx]
model_df_indexed = model_df.set_index("ModelID")
cancer_type_map  = {}
for cl in test_cell_lines:
    if cl in model_df_indexed.index:
        ct = model_df_indexed.loc[cl, "OncotreeLineage"]
        cancer_type_map[cl] = "Unknown" if pd.isna(ct) else str(ct)
    else:
        cancer_type_map[cl] = "Unknown"

tp_all = all_test_results[best_gnp_name]["preds"]
tl_all = all_test_results[best_gnp_name]["labels"]
tc_all = all_test_results[best_gnp_name]["cl_idx"]
tg_all = all_test_results[best_gnp_name]["g_idx"]

cancer_preds = {}; cancer_labels = {}
for pred, label, ci in zip(tp_all, tl_all, tc_all):
    ct = cancer_type_map.get(cell_lines[int(ci)], "Unknown")
    cancer_preds.setdefault(ct, []).append(pred)
    cancer_labels.setdefault(ct, []).append(label)

cancer_pearson = {
    ct: pearsonr(cancer_preds[ct], cancer_labels[ct])[0]
    for ct in cancer_preds if len(cancer_preds[ct]) >= 10}
cancer_pearson = dict(sorted(cancer_pearson.items(), key=lambda x: x[1], reverse=True))

log.info(f"\n  {'Cancer Type':<25} {'Pearson':>8} {'N':>6}")
for ct, p in cancer_pearson.items():
    log.info(f"  {ct:<25} {p:>8.4f} {len(cancer_preds[ct]):>6}")

if cancer_pearson:
    top_ct = list(cancer_pearson.keys())[:15]
    top_p  = [cancer_pearson[c] for c in top_ct]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_ct, top_p, color='steelblue')
    plt.ylabel("Pearson"); plt.title("Performance by Cancer Type (Top 15)")
    plt.xticks(rotation=45, ha='right'); plt.ylim(0, 1.0)
    for bar, val in zip(bars, top_p):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_cancer_type.png", dpi=150); plt.close()

# =============================================================================
# SYN vs ANT / SYNTHETIC LETHALITY ANALYSIS
# =============================================================================

gene_pred_scores = {}
for pred, gi in zip(tp_all, tg_all):
    gene_pred_scores.setdefault(genes[gi], []).append(pred)
avg_gene_pred = {g: np.mean(v) for g, v in gene_pred_scores.items()}

syn_scores = []; ant_scores = []; syn_pairs = []; ant_pairs = []
if len(cdns_df) > 0:
    for _, row in cdns_df.iterrows():
        ga = str(row.get('gene', row.iloc[0])).strip()
        gb = str(row.get('context_gene', row.iloc[1])).strip()
        ed = str(row.get('ed_type', row.iloc[2])).upper()
        sa = avg_gene_pred.get(ga); sb = avg_gene_pred.get(gb)
        if sa is not None and sb is not None:
            avg_pair = (sa + sb) / 2
            if 'SYN' in ed:
                syn_scores.append(avg_pair); syn_pairs.append((ga, gb, avg_pair))
            elif 'ANT' in ed:
                ant_scores.append(avg_pair); ant_pairs.append((ga, gb, avg_pair))
    if syn_scores and ant_scores:
        log.info(f"  SYN mean: {np.mean(syn_scores):.4f}  ANT mean: {np.mean(ant_scores):.4f}")
        plt.figure(figsize=(7, 5))
        plt.boxplot([syn_scores, ant_scores], labels=['SYN pairs', 'ANT pairs'],
                    patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
        plt.ylabel("Avg Predicted Dependency Score")
        plt.title("SYN vs ANT Gene Pairs"); plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/fig_syn_ant.png", dpi=150); plt.close()

if len(cdns_df) > 0 and ant_pairs:
    ant_sorted = sorted(ant_pairs, key=lambda x: x[2])
    log.info(f"\n  Top Synthetic Lethality Candidates (ANT pairs, lowest dep score):")
    for ga, gb, sc in ant_sorted[:15]:
        log.info(f"    {ga:<12} {gb:<12}  avg_dep={sc:.4f}")
    top_sl_names  = [f"{ga}-{gb}" for ga, gb, _ in ant_sorted[:15]]
    top_sl_scores = [sc for _, _, sc in ant_sorted[:15]]
    plt.figure(figsize=(12, 5))
    plt.bar(top_sl_names, top_sl_scores, color='coral')
    plt.ylabel("Avg Predicted Dependency Score")
    plt.title("Top Synthetic Lethality Candidates (ANT Gene Pairs)")
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=-0.5, color='red', linestyle='--', linewidth=1,
                label='Essential threshold (-0.5)')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_synthetic_lethality.png", dpi=150); plt.close()

# =============================================================================
# EXPLAINABILITY - ATTENTION HEATMAP
# =============================================================================

log.info("\n" + "=" * 60)
log.info("Explainability - Attention Heatmap...")

gene_avg_scores_mean = {}
for score, gi in zip(tp2, tg2):
    gene_avg_scores_mean.setdefault(genes[gi], []).append(score)
gene_avg_scores_mean = {g: np.mean(v) for g, v in gene_avg_scores_mean.items()}
top30_genes        = [g for g, _ in sorted(
    gene_avg_scores_mean.items(), key=lambda x: x[1])[:30]]
top30_gene_indices = [gene_to_idx[g] for g in top30_genes if g in gene_to_idx]

core_fig_model = unwrap(fig_model)
recorder = AttentionHeatmapRecorder(core_fig_model.attention, top30_gene_indices)
core_fig_model.eval()
with torch.no_grad():
    for ci_b, _, _ in test_loader:
        core_fig_model(ci_b.to(device), omics, edge_index, fig_ea, gene_pe)
attn_heatmap = recorder.get_mean_weights()
recorder.detach()

if attn_weights_arr is not None:
    sample_size = min(5000, len(attn_weights_arr))
    np.random.seed(RANDOM_SEED)
    idx_s      = np.random.choice(len(attn_weights_arr), sample_size, replace=False)
    pca        = PCA(n_components=2)
    pca_result = pca.fit_transform(attn_weights_arr[idx_s])
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                     c=tl2[idx_s], cmap='RdYlBu', alpha=0.5, s=1)
    plt.colorbar(sc, label='Actual Dependency Score')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA of Gene Attention Embeddings")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_pca_attention.png", dpi=150); plt.close()

gene_essential_scores = np.array([
    gene_avg_scores_mean.get(g, 0.0) for g in top30_genes])

cdns_gene_type = {}
if len(cdns_df) > 0:
    for _, row in cdns_df.iterrows():
        ga = str(row.get('gene', row.iloc[0])).strip()
        ed = str(row.get('ed_type', row.iloc[2])).upper()
        if ga not in cdns_gene_type:
            cdns_gene_type[ga] = 'SYN' if 'SYN' in ed else ('ANT' if 'ANT' in ed else 'None')

cdns_type_colors = {
    'SYN':  [0.2, 0.7, 0.3, 1.0],
    'ANT':  [1.0, 0.5, 0.1, 1.0],
    'None': [0.8, 0.8, 0.8, 1.0],
}
cdns_strip = np.array([
    cdns_type_colors.get(cdns_gene_type.get(g, 'None'), cdns_type_colors['None'])
    for g in top30_genes]).reshape(-1, 1, 4)

norm_ess   = plt.Normalize(vmin=gene_essential_scores.min(),
                            vmax=gene_essential_scores.max())
ess_colors = plt.cm.RdYlBu_r(norm_ess(gene_essential_scores))
ess_strip  = ess_colors.reshape(-1, 1, 4)

fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(1, 4, width_ratios=[0.06, 0.06, 1, 0.25], wspace=0.05)
ax_ess  = fig.add_subplot(gs[0])
ax_cdns = fig.add_subplot(gs[1])
ax_hm   = fig.add_subplot(gs[2])
ax_leg  = fig.add_subplot(gs[3])

ax_ess.imshow(ess_strip, aspect='auto', interpolation='nearest')
ax_ess.set_xticks([]); ax_ess.set_yticks(range(len(top30_genes)))
ax_ess.set_yticklabels(top30_genes, fontsize=9)
ax_ess.set_title('Dep\nScore', fontsize=8, pad=4)
ax_ess.tick_params(axis='y', length=0)
ax_cdns.imshow(cdns_strip, aspect='auto', interpolation='nearest')
ax_cdns.set_xticks([]); ax_cdns.set_yticks([])
ax_cdns.set_title('cdNS\nType', fontsize=8, pad=4)
im = ax_hm.imshow(attn_heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax_hm.set_xticks([]); ax_hm.set_yticks([])
ax_hm.set_xlabel("Attention Dimensions (256)", fontsize=10)
ax_hm.set_title("Gene Attention Heatmap - Top 30 Most Essential Genes", fontsize=11)
plt.colorbar(im, ax=ax_hm, label='Mean Attention Weight', fraction=0.03, pad=0.02)
ax_leg.axis('off')
from matplotlib.patches import Patch
leg1 = ax_leg.legend(handles=[
    Patch(facecolor=[0.2, 0.7, 0.3], label='SYN gene'),
    Patch(facecolor=[1.0, 0.5, 0.1], label='ANT gene'),
    Patch(facecolor=[0.8, 0.8, 0.8], label='No cdNS'),
], loc='upper center', fontsize=9, title='cdNS Type',
   title_fontsize=9, frameon=True, borderpad=1.0)
ax_leg.add_artist(leg1)
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm_ess); sm.set_array([])
cbar2 = plt.colorbar(sm, ax=ax_leg, fraction=0.3, pad=0.05,
                     shrink=0.5, anchor=(0.5, 0.0))
cbar2.set_label('Avg Predicted\nDependency Score', fontsize=8)
cbar2.ax.tick_params(labelsize=7)
for gi in range(len(top30_genes)):
    for ax_ in [ax_ess, ax_cdns, ax_hm]:
        ax_.axhline(y=gi + 0.5, color='white', linewidth=0.3, alpha=0.5)
plt.suptitle("Gene Attention Heatmap\nLeft=dep score | Middle=cdNS type",
             fontsize=10, y=1.01)
plt.savefig(f"{FIGURES_DIR}/fig_attention_heatmap.png", dpi=150,
            bbox_inches='tight'); plt.close()
log.info("Attention heatmap saved.")

# =============================================================================
# cdNS SCORE DISTRIBUTION
# =============================================================================

if len(cdns_df) > 0:
    syn_mask      = cdns_df['ed_type'].str.upper().str.contains('SYN')
    ant_mask      = cdns_df['ed_type'].str.upper().str.contains('ANT')
    syn_cdns_vals = cdns_df.loc[syn_mask, 'cdns'].astype(float).values
    ant_cdns_vals = cdns_df.loc[ant_mask, 'cdns'].astype(float).values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(min(syn_cdns_vals.min(), ant_cdns_vals.min()),
                       max(syn_cdns_vals.max(), ant_cdns_vals.max()), 30)
    axes[0].hist(syn_cdns_vals, bins=bins, alpha=0.6, color='steelblue',
                 label=f'SYN (n={len(syn_cdns_vals)})', density=True)
    axes[0].hist(ant_cdns_vals, bins=bins, alpha=0.6, color='coral',
                 label=f'ANT (n={len(ant_cdns_vals)})', density=True)
    axes[0].set_xlabel("cdNS Score"); axes[0].set_ylabel("Density")
    axes[0].set_title("cdNS Score Distribution - SYN vs ANT"); axes[0].legend()
    vp = axes[1].violinplot([syn_cdns_vals, ant_cdns_vals],
                             positions=[1, 2], showmeans=True, showmedians=True)
    vp['bodies'][0].set_facecolor('steelblue'); vp['bodies'][0].set_alpha(0.6)
    vp['bodies'][1].set_facecolor('coral');     vp['bodies'][1].set_alpha(0.6)
    axes[1].set_xticks([1, 2]); axes[1].set_xticklabels(['SYN', 'ANT'])
    axes[1].set_ylabel("cdNS Score"); axes[1].set_title("cdNS Score Violin Plot")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_cdns_distribution.png", dpi=150,
                bbox_inches='tight'); plt.close()

# =============================================================================
# PREDICTION ERROR HEATMAP
# =============================================================================

error_dict  = {}
for pred_v, label_v, ci, gene_i in zip(tp_all, tl_all, tc_all, tg_all):
    error_dict[(int(ci), int(gene_i))] = pred_v - label_v
test_cl_set = set(int(ci) for ci in tc_all)
gene_mae    = {}
for gi in range(num_genes):
    errs = [abs(error_dict[(ci, gi)]) for ci in test_cl_set if (ci, gi) in error_dict]
    if len(errs) >= 5:
        gene_mae[gi] = np.mean(errs)
top50_gene_idx = sorted(gene_mae, key=gene_mae.get, reverse=True)[:50]
top50_genes    = [genes[gi] for gi in top50_gene_idx]
test_cl_sorted = sorted(test_cl_set)
error_matrix   = np.full((len(test_cl_sorted), 50), np.nan)
for row, ci in enumerate(test_cl_sorted):
    for col, gi in enumerate(top50_gene_idx):
        if (ci, gi) in error_dict:
            error_matrix[row, col] = error_dict[(ci, gi)]
error_matrix_disp = np.nan_to_num(error_matrix, nan=0.0)

cl_cancer_labels = [cancer_type_map.get(cell_lines[ci], "Unknown")
                    for ci in test_cl_sorted]
sort_order    = sorted(range(len(cl_cancer_labels)), key=lambda i: cl_cancer_labels[i])
sorted_labels = [cl_cancer_labels[i] for i in sort_order]
sorted_errors = error_matrix_disp[sort_order, :]
unique_cancers   = sorted(set(sorted_labels))
cancer_colors    = plt.cm.tab20(np.linspace(0, 1, max(len(unique_cancers), 1)))
cancer_color_map = {ct: cancer_colors[i] for i, ct in enumerate(unique_cancers)}
color_strip      = np.array(
    [cancer_color_map[ct] for ct in sorted_labels]).reshape(-1, 1, 4)

fig, (ax_strip, ax, ax_leg) = plt.subplots(
    1, 3, figsize=(22, max(8, len(test_cl_sorted) // 8)),
    gridspec_kw={'width_ratios': [0.015, 1, 0.15]})
ax_strip.imshow(color_strip, aspect='auto', interpolation='nearest')
ax_strip.set_xticks([]); ax_strip.set_yticks([])
ax_strip.set_ylabel(f"Test Cell Lines (n={len(test_cl_sorted)}, sorted by cancer type)",
                    fontsize=9)
im = ax.imshow(sorted_errors, aspect='auto', cmap='RdBu_r',
               vmin=-1.0, vmax=1.0, interpolation='nearest')
ax.set_xticks(range(50))
ax.set_xticklabels(top50_genes, rotation=90, fontsize=6)
ax.set_yticks([])
ax.set_title("Prediction Error Heatmap - Top-50 Highest-Error Genes\n"
             "Blue=under-prediction | Red=over-prediction", fontsize=10)
plt.colorbar(im, ax=ax, label='Prediction Error (pred - actual)',
             fraction=0.02, pad=0.01)
ax_leg.axis('off')
ax_leg.legend(
    [plt.Rectangle((0, 0), 1, 1, facecolor=cancer_color_map[ct],
                   edgecolor='grey', linewidth=0.5) for ct in unique_cancers],
    unique_cancers, loc='center left', fontsize=6,
    title='Cancer Type', title_fontsize=7, frameon=True, ncol=1)
cumsum = 0
for ct, cnt in zip(*np.unique(sorted_labels, return_counts=True)):
    cumsum += cnt
    if cumsum < len(sorted_labels):
        for ax_ in [ax, ax_strip]:
            ax_.axhline(y=cumsum - 0.5, color='black', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_error_heatmap.png", dpi=150,
            bbox_inches='tight'); plt.close()
log.info("Prediction error heatmap saved.")

# =============================================================================
# GENE NETWORK SUBGRAPH
# =============================================================================

if HAS_NX:
    top10_genes   = [g for g, _ in sorted(
        gene_avg_scores_mean.items(), key=lambda x: x[1])[:10]]
    top10_idx_set = {gene_to_idx[g] for g in top10_genes if g in gene_to_idx}
    ei_cpu        = edge_index.cpu().numpy()
    subgraph_nodes = set(top10_idx_set)
    for i in range(ei_cpu.shape[1]):
        if ei_cpu[0, i] in top10_idx_set:
            subgraph_nodes.add(ei_cpu[1, i])
    if len(subgraph_nodes) > 80:
        nbrs = subgraph_nodes - top10_idx_set
        nbrs_sorted = sorted(nbrs, key=lambda ni: abs(
            gene_avg_scores_mean.get(genes[ni], 0.0)), reverse=True)[:70]
        subgraph_nodes = top10_idx_set | set(nbrs_sorted)
    G_sub = nx.Graph()
    for ni in subgraph_nodes:
        G_sub.add_node(ni, gene=genes[ni],
                       score=gene_avg_scores_mean.get(genes[ni], 0.0))
    for i in range(ei_cpu.shape[1]):
        sn, dn = ei_cpu[0, i], ei_cpu[1, i]
        if sn in subgraph_nodes and dn in subgraph_nodes and sn < dn:
            G_sub.add_edge(sn, dn)
    node_scores  = np.array([G_sub.nodes[n]['score'] for n in G_sub.nodes()])
    node_labels  = {n: genes[n] if n in top10_idx_set else '' for n in G_sub.nodes()}
    node_sizes   = [300 if n in top10_idx_set else 60 for n in G_sub.nodes()]
    norm     = mcolors.TwoSlopeNorm(
        vmin=node_scores.min(), vcenter=0.0, vmax=max(node_scores.max(), 0.01))
    cmap     = plt.cm.RdYlBu_r
    node_col = [cmap(norm(s)) for s in node_scores]
    pos = nx.spring_layout(G_sub, seed=RANDOM_SEED, k=0.5)
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_col,
                           node_size=node_sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_labels(G_sub, pos, node_labels, font_size=8, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Avg Predicted Dependency Score')
    ax.set_title("Gene Network Subgraph - Top-10 Essential Genes + Neighbours",
                 fontsize=12)
    ax.axis('off'); plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_gene_network.png", dpi=150); plt.close()
    log.info("Gene network subgraph saved.")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

log.info("\n" + "=" * 70)
log.info("  GeneNetPred run complete")
log.info("=" * 70)
log.info(f"  Total Training Time : {total_training_time/3600:.2f} hours")
log.info(f"  Best GNP Model      : {best_gnp_name}  r={pearson:.4f}")
log.info(f"  DepGPS variants     : {len(DEPGPS_VARIANTS)} x 2 (cdNS on/off) = 8 runs")
log.info(f"  DeepDEP variants    : 2 (cdNS on/off)")
log.info(f"  Multi-seed seeds    : {len(SEEDS)}")
log.info(f"  Results             : {OUTPUT_DIR}")
log.info(f"  Figures             : {FIGURES_DIR}")
log.info(f"  Checkpoints         : {CHECKPOINT_DIR}")
log.info(f"  Log                 : {log_path}")
log.info(f"  Completed at        : {datetime.now()}")
log.info("=" * 70)
