"""
Tests for Anomaly Detection in Network Traffic
"""
import numpy as np
import pytest
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from detector import (
    generate_traffic_data,
    preprocess,
    preprocess_engineered,
    engineer_features,
    train_isolation_forest,
    train_lof,
    train_dbscan,
    zscore_anomaly,
    ensemble_vote,
    score_to_threat,
    _hash_row,
    FEATURE_COLS,
    ENG_COLS,
)


def test_data_generation():
    df = generate_traffic_data(n_normal=100, n_anomaly=20)
    assert len(df) == 120
    assert "label" in df.columns
    assert "traffic_id" in df.columns
    assert set(df["label"].unique()) >= {"normal"}


def test_feature_engineering():
    df = generate_traffic_data(n_normal=50, n_anomaly=10)
    df_eng = engineer_features(df)
    assert "bytes_ratio"   in df_eng.columns
    assert "is_off_hours"  in df_eng.columns
    assert "bytes_per_pkt" in df_eng.columns
    assert df_eng["is_off_hours"].isin([0, 1]).all()


def test_preprocess_shape():
    df = generate_traffic_data(n_normal=100, n_anomaly=20)
    X_raw, X_scaled, scaler = preprocess(df)
    assert X_scaled.shape == (120, len(FEATURE_COLS))
    assert abs(X_scaled.mean()) < 0.2


def test_isolation_forest_output():
    df = generate_traffic_data(n_normal=200, n_anomaly=20)
    _, X_scaled, _ = preprocess_engineered(df)
    model, labels, scores = train_isolation_forest(X_scaled, contamination=0.09)
    assert labels.shape == (220,)
    assert set(labels).issubset({0, 1})
    assert labels.sum() > 0


def test_lof_output():
    df = generate_traffic_data(n_normal=200, n_anomaly=20)
    _, X_scaled, _ = preprocess_engineered(df)
    _, labels, scores = train_lof(X_scaled, contamination=0.09)
    assert labels.shape == (220,)
    assert set(labels).issubset({0, 1})


def test_dbscan_output():
    df = generate_traffic_data(n_normal=200, n_anomaly=20)
    _, X_scaled, _ = preprocess_engineered(df)
    _, labels, _ = train_dbscan(X_scaled)
    assert labels.shape == (220,)


def test_zscore_output():
    df = generate_traffic_data(n_normal=200, n_anomaly=20)
    _, X_scaled, _ = preprocess_engineered(df)
    labels, scores = zscore_anomaly(X_scaled)
    assert labels.shape == (220,)
    assert scores.shape == (220,)


def test_ensemble_majority():
    a = np.array([1, 0, 1, 0, 1])
    b = np.array([1, 0, 0, 1, 1])
    c = np.array([0, 0, 1, 0, 1])
    d = np.array([0, 0, 0, 0, 0])
    result = ensemble_vote({"A":a,"B":b,"C":c,"D":d})
    # [2,0,2,1,3] → >=2 → [1,0,1,0,1]
    np.testing.assert_array_equal(result, [1, 0, 1, 0, 1])


def test_threat_levels():
    assert score_to_threat(0, 0.1, {}) == "NORMAL"
    assert score_to_threat(1, 0.5, {"failed_logins": 50}) == "CRITICAL"
    assert score_to_threat(1, 0.5, {"bytes_sent": 200000}) == "HIGH"
    assert score_to_threat(1, 2.0, {}) == "HIGH"
    assert score_to_threat(1, 0.5, {"unique_ports": 80}) == "MEDIUM"
    assert score_to_threat(1, 0.5, {}) == "LOW"


def test_hash_deterministic():
    r = dict(bytes_sent=1000, failed_logins=2)
    assert _hash_row(r) == _hash_row(r)
