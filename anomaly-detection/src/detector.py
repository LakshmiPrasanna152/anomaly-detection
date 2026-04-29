"""
╔══════════════════════════════════════════════════════════╗
║   Anomaly Detection in Network Traffic                   ║
║   Domain: Cybersecurity / System Monitoring              ║
║   Algorithms: Isolation Forest | LOF | DBSCAN | Z-Score  ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings, json, os, hashlib
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import box
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule

warnings.filterwarnings("ignore")
console = Console()

SNAPSHOT_PATH = "models/snapshots.json"
REPORT_DIR    = "reports/figures"

# Threat severity mapping
THREAT_LEVELS = {
    "NORMAL":         ("🟢", "green",   "Normal traffic — no action needed"),
    "LOW":            ("🟡", "yellow",  "Mild anomaly — log and monitor"),
    "MEDIUM":         ("🟠", "dark_orange","Suspicious — investigate within 24h"),
    "HIGH":           ("🔴", "red",     "High-risk — escalate immediately"),
    "CRITICAL":       ("💀", "bold red","CRITICAL — block & alert SOC team"),
}

# ─── Utilities ─────────────────────────────────────────────────────────────────

def _hash_row(row: dict) -> str:
    return hashlib.md5(json.dumps(row, sort_keys=True).encode()).hexdigest()[:10]

def load_snapshots() -> dict:
    if os.path.exists(SNAPSHOT_PATH):
        with open(SNAPSHOT_PATH) as f:
            return json.load(f)
    return {}

def save_snapshot(key: str, payload: dict):
    snaps = load_snapshots()
    snaps[key] = payload
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snaps, f, indent=2)

# ─── Synthetic Network Traffic Generator ───────────────────────────────────────

def generate_traffic_data(n_normal: int = 800, n_anomaly: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = n_normal + n_anomaly

    # ── Normal traffic (HTTP/HTTPS browsing, DNS, NTP etc.) ──
    normal = pd.DataFrame({
        "bytes_sent":      rng.normal(5000,  1500,  n_normal).clip(100, 15000),
        "bytes_recv":      rng.normal(30000, 8000,  n_normal).clip(500, 80000),
        "duration_ms":     rng.normal(200,   80,    n_normal).clip(10, 800),
        "packet_count":    rng.normal(40,    12,    n_normal).clip(2, 150).astype(int),
        "unique_ports":    rng.integers(1, 5,        n_normal),
        "failed_logins":   rng.integers(0, 2,        n_normal),
        "icmp_ratio":      rng.uniform(0,   0.05,   n_normal),
        "syn_ratio":       rng.uniform(0.1, 0.4,    n_normal),
        "protocol":        rng.choice(["TCP","UDP","HTTPS","DNS"], n_normal, p=[0.5,0.2,0.25,0.05]),
        "hour":            rng.integers(8, 20,       n_normal),   # business hours
        "label":           ["normal"] * n_normal,
    })

    # ── Anomaly sub-types ──
    n1, n2, n3, n4 = n_anomaly//4, n_anomaly//4, n_anomaly//4, n_anomaly - 3*(n_anomaly//4)

    # Port scan
    portscan = pd.DataFrame({
        "bytes_sent":      rng.normal(200,   50,    n1).clip(50, 800),
        "bytes_recv":      rng.normal(300,   100,   n1).clip(50, 1000),
        "duration_ms":     rng.normal(5000,  1000,  n1).clip(1000, 20000),
        "packet_count":    rng.normal(500,   100,   n1).clip(100, 2000).astype(int),
        "unique_ports":    rng.integers(50, 200,    n1),
        "failed_logins":   rng.integers(0, 3,       n1),
        "icmp_ratio":      rng.uniform(0.3, 0.9,   n1),
        "syn_ratio":       rng.uniform(0.7, 1.0,   n1),
        "protocol":        rng.choice(["TCP","ICMP"], n1),
        "hour":            rng.integers(0, 6,        n1),
        "label":           ["port_scan"] * n1,
    })
    # Data exfiltration
    exfil = pd.DataFrame({
        "bytes_sent":      rng.normal(200000, 50000, n2).clip(50000, 500000),
        "bytes_recv":      rng.normal(5000,   2000,  n2).clip(500, 20000),
        "duration_ms":     rng.normal(30000,  8000,  n2).clip(5000, 120000),
        "packet_count":    rng.normal(800,    200,   n2).clip(100, 3000).astype(int),
        "unique_ports":    rng.integers(1, 3,         n2),
        "failed_logins":   rng.integers(0, 2,         n2),
        "icmp_ratio":      rng.uniform(0, 0.02,      n2),
        "syn_ratio":       rng.uniform(0.05, 0.2,    n2),
        "protocol":        rng.choice(["TCP","HTTPS"], n2),
        "hour":            rng.integers(1, 5,          n2),
        "label":           ["data_exfil"] * n2,
    })
    # Brute force
    brute = pd.DataFrame({
        "bytes_sent":      rng.normal(800,   200,   n3).clip(100, 3000),
        "bytes_recv":      rng.normal(600,   200,   n3).clip(100, 2000),
        "duration_ms":     rng.normal(100,   30,    n3).clip(10, 500),
        "packet_count":    rng.normal(60,    15,    n3).clip(10, 200).astype(int),
        "unique_ports":    rng.integers(1, 3,        n3),
        "failed_logins":   rng.integers(10, 100,    n3),
        "icmp_ratio":      rng.uniform(0, 0.03,     n3),
        "syn_ratio":       rng.uniform(0.4, 0.7,    n3),
        "protocol":        rng.choice(["TCP","SSH"], n3),
        "hour":            rng.integers(0, 24,       n3),
        "label":           ["brute_force"] * n3,
    })
    # DDoS
    ddos = pd.DataFrame({
        "bytes_sent":      rng.normal(100,   30,    n4).clip(20, 500),
        "bytes_recv":      rng.normal(100,   30,    n4).clip(20, 500),
        "duration_ms":     rng.normal(5,     2,     n4).clip(1, 30),
        "packet_count":    rng.normal(2000,  500,   n4).clip(500, 10000).astype(int),
        "unique_ports":    rng.integers(1, 4,        n4),
        "failed_logins":   rng.integers(0, 3,        n4),
        "icmp_ratio":      rng.uniform(0.5, 1.0,    n4),
        "syn_ratio":       rng.uniform(0.8, 1.0,    n4),
        "protocol":        rng.choice(["UDP","ICMP","TCP"], n4),
        "hour":            rng.integers(0, 24,       n4),
        "label":           ["ddos"] * n4,
    })

    df = pd.concat([normal, portscan, exfil, brute, ddos], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["traffic_id"] = [f"PKT-{10000+i}" for i in range(len(df))]
    df["timestamp"]  = pd.date_range("2025-01-01", periods=len(df), freq="5min")
    return df

# ─── Preprocessing ─────────────────────────────────────────────────────────────

FEATURE_COLS = ["bytes_sent","bytes_recv","duration_ms","packet_count",
                "unique_ports","failed_logins","icmp_ratio","syn_ratio","hour"]

def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler

# ─── Feature Engineering (derived features) ────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bytes_ratio"]     = df["bytes_sent"] / (df["bytes_recv"] + 1)
    df["bytes_per_pkt"]   = (df["bytes_sent"] + df["bytes_recv"]) / (df["packet_count"] + 1)
    df["is_off_hours"]    = ((df["hour"] < 7) | (df["hour"] > 22)).astype(int)
    df["port_diversity"]  = np.log1p(df["unique_ports"])
    return df

ENG_COLS = FEATURE_COLS + ["bytes_ratio","bytes_per_pkt","is_off_hours","port_diversity"]

def preprocess_engineered(df: pd.DataFrame):
    df_eng = engineer_features(df)
    X = df_eng[ENG_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler

# ─── Models ────────────────────────────────────────────────────────────────────

def train_isolation_forest(X_scaled, contamination=0.09):
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42, n_jobs=-1)
    preds = model.fit_predict(X_scaled)           # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)    # lower = more anomalous
    labels = np.where(preds == -1, 1, 0)          # 1 = anomaly
    return model, labels, scores

def train_lof(X_scaled, contamination=0.09):
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
    preds = model.fit_predict(X_scaled)
    scores = -model.negative_outlier_factor_
    labels = np.where(preds == -1, 1, 0)
    return model, labels, scores

def train_dbscan(X_scaled, eps=1.2, min_samples=8):
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    cluster_labels = model.fit_predict(X_scaled)
    labels = (cluster_labels == -1).astype(int)   # noise = anomaly
    scores = np.zeros(len(X_scaled))
    for i, lbl in enumerate(cluster_labels):
        if lbl == -1:
            scores[i] = 2.0
    return model, labels, scores

def zscore_anomaly(X_scaled, threshold=3.0):
    z = np.abs(stats.zscore(X_scaled))
    labels = (z.max(axis=1) > threshold).astype(int)
    scores = z.max(axis=1)
    return labels, scores

# ─── Ensemble Voting ───────────────────────────────────────────────────────────

def ensemble_vote(all_labels: dict) -> np.ndarray:
    stack  = np.stack(list(all_labels.values()), axis=1)
    votes  = stack.sum(axis=1)
    # majority: flag if ≥2 models agree
    return (votes >= 2).astype(int)

def score_to_threat(anomaly_flag: int, score: float, row: dict) -> str:
    if anomaly_flag == 0:
        return "NORMAL"
    # Escalate based on features
    if row.get("failed_logins", 0) > 20:
        return "CRITICAL"
    if row.get("bytes_sent", 0) > 100000:
        return "HIGH"
    if score > 1.5:
        return "HIGH"
    if row.get("unique_ports", 0) > 50:
        return "MEDIUM"
    return "LOW"

# ─── Metrics Table ─────────────────────────────────────────────────────────────

def print_model_comparison(results: dict, true_labels: np.ndarray):
    table = Table(title="🔬 Model Detection Comparison", box=box.ROUNDED,
                  header_style="bold cyan", show_lines=True)
    table.add_column("Model",           style="bold", width=20)
    table.add_column("Flagged",         justify="center")
    table.add_column("True Anomalies",  justify="center")
    table.add_column("Precision",       justify="center")
    table.add_column("Recall",          justify="center")
    table.add_column("F1",              justify="center")

    n_true = true_labels.sum()
    for name, (labels, _scores) in results.items():
        flagged = labels.sum()
        # Align with true labels
        tp = ((labels == 1) & (true_labels == 1)).sum()
        fp = ((labels == 1) & (true_labels == 0)).sum()
        fn = ((labels == 0) & (true_labels == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        color = "green" if f1 > 0.7 else ("yellow" if f1 > 0.4 else "red")
        table.add_row(
            name, str(flagged), str(n_true),
            f"[{color}]{prec:.3f}[/]",
            f"[{color}]{rec:.3f}[/]",
            f"[bold {color}]{f1:.3f}[/]",
        )
    console.print(table)

# ─── Threat Summary Table ───────────────────────────────────────────────────────

def print_threat_summary(df: pd.DataFrame, anomaly_labels: np.ndarray, threat_col: str = "threat_level"):
    df2 = df.copy()
    df2["is_anomaly"] = anomaly_labels
    summary = df2[df2["is_anomaly"] == 1]["label"].value_counts().reset_index()
    summary.columns = ["Attack Type", "Count"]

    table = Table(title="🚨 Detected Threat Summary", box=box.ROUNDED,
                  header_style="bold red", show_lines=True)
    table.add_column("Attack Type",  style="bold", width=18)
    table.add_column("Count",        justify="center")
    table.add_column("% of Anomalies", justify="center")
    table.add_column("Risk",         width=30)

    risk_map = {
        "normal":      ("🟢", "Baseline — investigate context"),
        "port_scan":   ("🟡", "Reconnaissance activity"),
        "data_exfil":  ("💀", "CRITICAL — data breach risk"),
        "brute_force": ("🔴", "Credential attack in progress"),
        "ddos":        ("🔴", "Service disruption attack"),
    }
    total = summary["Count"].sum()
    for _, row in summary.iterrows():
        atype = row["Attack Type"]
        icon, risk = risk_map.get(atype, ("❓","Unknown"))
        pct = f"{row['Count']/total*100:.1f}%"
        table.add_row(f"{icon} {atype}", str(row["Count"]), pct, risk)
    console.print(table)

# ─── Visualisations ────────────────────────────────────────────────────────────

def _pca2(X): return PCA(n_components=2, random_state=42).fit_transform(X)

DARK_BG   = "#0d0d1a"
PANEL_BG  = "#12122a"
GRID_COL  = "#1e1e40"

def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.set_xlabel(ax.get_xlabel(), color="#aaaacc", fontsize=9)
    ax.set_ylabel(ax.get_ylabel(), color="#aaaacc", fontsize=9)
    if title: ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)

def plot_anomaly_scatter(X_scaled, labels, title, filepath):
    coords = _pca2(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

    # Scatter
    mask_n = labels == 0
    mask_a = labels == 1
    axes[0].scatter(coords[mask_n,0], coords[mask_n,1],
                    c="#3ecf8e", alpha=0.55, s=18, label="Normal", edgecolors="none")
    axes[0].scatter(coords[mask_a,0], coords[mask_a,1],
                    c="#e74c3c", alpha=0.85, s=45, label="Anomaly", marker="X", zorder=5)
    _style_ax(axes[0], f"{title} — PCA Projection")
    axes[0].legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white", framealpha=0.7)

    # Timeline anomaly density
    idx = np.arange(len(labels))
    axes[1].fill_between(idx, labels.astype(float), alpha=0.5, color="#e74c3c", label="Anomaly flag")
    axes[1].plot(idx, labels.astype(float), color="#e74c3c", linewidth=0.6, alpha=0.6)
    _style_ax(axes[1], "Anomaly Flag Timeline")
    axes[1].set_xlabel("Packet Index", color="#aaaacc")
    axes[1].set_ylabel("Anomaly (1=Yes)", color="#aaaacc")

    fig.suptitle(f"Network Anomaly Detection | {title}", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

def plot_feature_heatmap(df: pd.DataFrame, labels: np.ndarray, filepath: str):
    df2 = df[FEATURE_COLS].copy()
    df2["is_anomaly"] = labels
    profile = df2.groupby("is_anomaly")[FEATURE_COLS].mean()
    profile.index = ["Normal","Anomaly"]
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    sns.heatmap(profile_norm.T, annot=profile.T.round(1), fmt=".1f",
                cmap="RdYlGn_r", linewidths=0.5, ax=ax,
                cbar_kws={"shrink":0.8}, annot_kws={"size":9,"color":"white"})
    ax.set_title("Feature Profile: Normal vs Anomaly (Normalized)", color="white", pad=10)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

def plot_score_distribution(scores_dict: dict, filepath: str):
    fig, axes = plt.subplots(1, len(scores_dict), figsize=(5*len(scores_dict), 4),
                             facecolor=DARK_BG)
    if len(scores_dict) == 1: axes = [axes]
    for ax, (name, scores) in zip(axes, scores_dict.items()):
        ax.hist(scores, bins=50, color="#7c5cbf", edgecolor="none", alpha=0.8)
        _style_ax(ax, f"{name}\nAnomaly Score Distribution")
        ax.set_xlabel("Score", color="#aaaacc")
        ax.set_ylabel("Count", color="#aaaacc")
    fig.suptitle("Anomaly Score Distributions", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

def plot_before_after(old_snap: dict, new_snap: dict, filepath: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    titles  = ["⬅  Previous Packet", "➡  Current Packet"]
    snaps   = [old_snap, new_snap]

    for ax, snap, ttl in zip(axes, snaps, titles):
        X      = np.array(snap["X_scaled"])
        labels = np.array(snap["labels"])
        coords = _pca2(X)
        mask_n = labels == 0
        mask_a = labels == 1
        ax.scatter(coords[mask_n,0], coords[mask_n,1],
                   c="#3ecf8e", alpha=0.4, s=14, label="Normal", edgecolors="none")
        ax.scatter(coords[mask_a,0], coords[mask_a,1],
                   c="#e74c3c", alpha=0.75, s=35, marker="X", label="Anomaly", zorder=5)
        # Highlight the single new packet
        pkt = np.array(snap["input_scaled"]).reshape(1,-1)
        pkt_pca = PCA(n_components=2, random_state=42).fit(X).transform(pkt)
        color = "#e74c3c" if snap["is_anomaly"] else "#3ecf8e"
        marker= "X"       if snap["is_anomaly"] else "o"
        ax.scatter(pkt_pca[0,0], pkt_pca[0,1], c=color, s=200, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=10, label="This packet")
        _style_ax(ax, ttl)
        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.7)

    threat_old = old_snap["threat_level"]
    threat_new = new_snap["threat_level"]
    fig.suptitle(
        f"🔄 Before/After  |  {threat_old} → {threat_new}",
        color="white", fontsize=13, y=1.01
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

# ─── Single-Packet Prediction ──────────────────────────────────────────────────

def predict_packet(iforest_model, scaler, row: dict, eng: bool = True) -> tuple:
    base = {k: row[k] for k in FEATURE_COLS}
    df_row = pd.DataFrame([base])
    if eng:
        df_row = engineer_features(df_row)
        cols = ENG_COLS
    else:
        cols = FEATURE_COLS
    X_raw   = df_row[cols].values
    X_sc    = scaler.transform(X_raw)
    pred    = iforest_model.predict(X_sc)[0]          # -1 anomaly, 1 normal
    score   = -iforest_model.decision_function(X_sc)[0]  # higher = more anomalous
    is_anom = 1 if pred == -1 else 0
    return is_anom, float(score), X_sc[0].tolist()

# ─── Interactive Loop ──────────────────────────────────────────────────────────

def interactive_loop(df: pd.DataFrame, X_scaled: np.ndarray,
                     ensemble_labels: np.ndarray, iforest_model, scaler):
    snaps   = load_snapshots()
    run_id  = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    console.print(Panel.fit(
        "[bold red]🛡️  Network Packet Anomaly Inspector[/]\n"
        "[dim]Enter packet features → instant threat classification + before/after comparison[/]",
        border_style="red", padding=(1,4)
    ))

    while True:
        console.rule("[bold yellow]📡 New Packet Entry")
        try:
            bytes_sent    = float(Prompt.ask("  Bytes Sent",             default="5200"))
            bytes_recv    = float(Prompt.ask("  Bytes Received",         default="28000"))
            duration_ms   = float(Prompt.ask("  Duration (ms)",          default="210"))
            packet_count  = int(Prompt.ask  ("  Packet Count",           default="38"))
            unique_ports  = int(Prompt.ask  ("  Unique Ports Accessed",  default="2"))
            failed_logins = int(Prompt.ask  ("  Failed Logins",          default="0"))
            icmp_ratio    = float(Prompt.ask("  ICMP Ratio (0.0–1.0)",   default="0.02"))
            syn_ratio     = float(Prompt.ask("  SYN Ratio  (0.0–1.0)",   default="0.25"))
            hour          = int(Prompt.ask  ("  Hour of Day (0–23)",      default="14"))
        except (ValueError, KeyboardInterrupt):
            console.print("[red]Invalid input — try again.[/]")
            continue

        row = dict(bytes_sent=bytes_sent, bytes_recv=bytes_recv,
                   duration_ms=duration_ms, packet_count=packet_count,
                   unique_ports=unique_ports, failed_logins=failed_logins,
                   icmp_ratio=icmp_ratio, syn_ratio=syn_ratio, hour=hour)

        row_id   = _hash_row(row)
        is_anom, score, input_scaled = predict_packet(iforest_model, scaler, row, eng=True)
        threat   = score_to_threat(is_anom, score, row)
        icon, color, advice = THREAT_LEVELS[threat]

        # ── Result Panel ──
        status_text = "⚠  ANOMALY DETECTED" if is_anom else "✅ NORMAL TRAFFIC"
        console.print(Panel(
            f"[bold white]Status     :[/] [bold {'red' if is_anom else 'green'}]{status_text}[/]\n"
            f"[bold white]Threat     :[/] {icon}  [{color}][bold]{threat}[/][/]\n"
            f"[bold white]Score      :[/] {score:.4f}  [dim](higher = more suspicious)[/]\n"
            f"[bold white]Advice     :[/] [dim]{advice}[/]\n"
            f"[bold white]Entry ID   :[/] [dim]{row_id}[/]",
            title=f"[bold {'red' if is_anom else 'green'}]🔍 Detection Result[/]",
            border_style=color if color != "bold red" else "red"
        ))

        new_snap = dict(
            input=row, is_anomaly=is_anom, score=score,
            threat_level=threat, input_scaled=input_scaled,
            X_scaled=X_scaled.tolist(), labels=ensemble_labels.tolist(),
            timestamp=datetime.now().isoformat()
        )

        # ── Before/After Comparison ──
        if snaps:
            last_key = list(snaps.keys())[-1]
            old_snap = snaps[last_key]
            old_row  = old_snap["input"]

            table = Table(title="🔄 Delta vs Previous Packet", box=box.SIMPLE_HEAVY,
                          header_style="bold magenta", show_lines=True)
            table.add_column("Feature",         style="bold", width=22)
            table.add_column("Previous",        justify="right", width=16)
            table.add_column("Current",         justify="right", width=16)
            table.add_column("Δ Change",        justify="right", width=14)

            fields = {
                "Bytes Sent":    "bytes_sent",
                "Bytes Recv":    "bytes_recv",
                "Duration (ms)": "duration_ms",
                "Packet Count":  "packet_count",
                "Unique Ports":  "unique_ports",
                "Failed Logins": "failed_logins",
                "ICMP Ratio":    "icmp_ratio",
                "SYN Ratio":     "syn_ratio",
                "Hour":          "hour",
            }
            for label_f, key in fields.items():
                prev = old_row.get(key, 0)
                curr = row.get(key, 0)
                d    = curr - prev
                if abs(d) < 1e-9:   d_str = "[dim]—[/]"
                elif d > 0:         d_str = f"[red]+{d:.2f}[/]"
                else:               d_str = f"[green]{d:.2f}[/]"
                table.add_row(label_f, f"{prev:.2f}", f"{curr:.2f}", d_str)

            old_icon, old_color, _ = THREAT_LEVELS[old_snap["threat_level"]]
            new_icon, new_color, _ = THREAT_LEVELS[threat]
            changed = old_snap["threat_level"] != threat
            table.add_row(
                "[bold]THREAT LEVEL[/]",
                f"{old_icon} {old_snap['threat_level']}",
                f"{new_icon} {threat}",
                "[bold yellow]⬆ ESCALATED[/]" if changed and is_anom
                else ("[bold green]⬇ CLEARED[/]"  if changed and not is_anom
                else  "[dim]unchanged[/]")
            )
            table.add_row(
                "[bold]ANOMALY SCORE[/]",
                f"{old_snap['score']:.4f}",
                f"{score:.4f}",
                f"[{'red' if score > old_snap['score'] else 'green'}]"
                f"{score - old_snap['score']:+.4f}[/]"
            )
            console.print(table)

            fig_path = f"{REPORT_DIR}/before_after_{run_id}.png"
            plot_before_after(old_snap, new_snap, fig_path)
            console.print(f"  [dim]📊 Before/After plot saved → {fig_path}[/]")

        save_snapshot(run_id, new_snap)
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not Confirm.ask("\n  [bold]Inspect another packet?[/]", default=True):
            console.print("[bold cyan]Session saved. Stay secure! 🔒[/]")
            break

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold red]🛡️  Anomaly Detection in Network Traffic[/]\n"
        "[dim]Isolation Forest | LOF | DBSCAN | Z-Score | Ensemble | Before-After[/]\n"
        "[dim]Domain: Cybersecurity / System Monitoring[/]",
        border_style="red", padding=(1,6)
    ))

    # ── 1. Data ──
    console.rule("[bold]Step 1 — Synthetic Traffic Generation")
    df = generate_traffic_data(n_normal=800, n_anomaly=80)
    n_anom = (df["label"] != "normal").sum()
    console.print(f"  [green]✓[/] {len(df)} packets  |  {n_anom} labelled anomalies  |  {df['label'].nunique()} attack types")
    df.to_csv("data/network_traffic.csv", index=False)

    # ── 2. Feature engineering + preprocess ──
    console.rule("[bold]Step 2 — Feature Engineering & Preprocessing")
    df_eng  = engineer_features(df)
    X_eng   = df_eng[ENG_COLS]
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)
    true_labels = (df["label"] != "normal").astype(int).values
    console.print(f"  [green]✓[/] {len(ENG_COLS)} features  (base: {len(FEATURE_COLS)} + engineered: {len(ENG_COLS)-len(FEATURE_COLS)})")

    # ── 3. Train all models ──
    console.rule("[bold]Step 3 — Model Training")
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  BarColumn(), transient=True) as prog:
        t = prog.add_task("Training Isolation Forest...", total=None)
        iforest_model, if_labels, if_scores = train_isolation_forest(X_scaled)
        prog.update(t, description="Training LOF...")
        _lof_model,   lof_labels, lof_scores = train_lof(X_scaled)
        prog.update(t, description="Training DBSCAN...")
        _dbscan_model, db_labels, db_scores  = train_dbscan(X_scaled)
        prog.update(t, description="Z-Score baseline...")
        zs_labels, zs_scores                 = zscore_anomaly(X_scaled)
    console.print("  [green]✓[/] All 4 models trained")

    # ── 4. Ensemble ──
    console.rule("[bold]Step 4 — Ensemble Voting (majority ≥2/4)")
    all_labels     = {"IsolationForest": if_labels, "LOF": lof_labels,
                      "DBSCAN": db_labels, "ZScore": zs_labels}
    ensemble_labels = ensemble_vote(all_labels)
    console.print(f"  [green]✓[/] Ensemble flagged [red]{ensemble_labels.sum()}[/] anomalies  "
                  f"({ensemble_labels.sum()/len(ensemble_labels)*100:.1f}% of traffic)")

    # ── 5. Metrics ──
    console.rule("[bold]Step 5 — Model Comparison")
    results_metrics = {
        "Isolation Forest": (if_labels,  if_scores),
        "LOF":              (lof_labels, lof_scores),
        "DBSCAN":           (db_labels,  db_scores),
        "Z-Score":          (zs_labels,  zs_scores),
        "Ensemble":         (ensemble_labels, if_scores),
    }
    print_model_comparison(results_metrics, true_labels)

    # ── 6. Threat summary ──
    console.rule("[bold]Step 6 — Threat Breakdown")
    print_threat_summary(df, ensemble_labels)

    # ── 7. Visualisations ──
    console.rule("[bold]Step 7 — Generating Reports")
    os.makedirs(REPORT_DIR, exist_ok=True)
    plot_anomaly_scatter(X_scaled, if_labels,
                         "Isolation Forest", f"{REPORT_DIR}/iforest_scatter.png")
    plot_anomaly_scatter(X_scaled, ensemble_labels,
                         "Ensemble",         f"{REPORT_DIR}/ensemble_scatter.png")
    plot_feature_heatmap(df, ensemble_labels, f"{REPORT_DIR}/feature_heatmap.png")
    plot_score_distribution(
        {"Isolation Forest": if_scores, "LOF": lof_scores, "Z-Score": zs_scores},
        f"{REPORT_DIR}/score_distributions.png"
    )
    console.print(f"  [green]✓[/] All plots → {REPORT_DIR}/")

    # ── 8. Export ──
    df["anomaly_ensemble"] = ensemble_labels
    df["if_score"]         = if_scores
    df.to_csv("data/traffic_with_anomalies.csv", index=False)
    console.print("  [green]✓[/] Exported → data/traffic_with_anomalies.csv")

    # ── 9. Interactive ──
    console.rule("[bold]Step 8 — Interactive Packet Inspector")
    interactive_loop(df, X_scaled, ensemble_labels, iforest_model, scaler)


if __name__ == "__main__":
    main()
