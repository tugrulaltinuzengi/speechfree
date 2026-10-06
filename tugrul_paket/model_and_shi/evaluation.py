import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_within_session_splits(X_session, y_session, n_splits=5):
    """
    Protokol A: Within-session (5-fold CV)
    Tek bir oturumun verisini 5'e böler. %80 Train, %20 Test.
    Döndürür: Generator yields (train_idx, test_idx)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return skf.split(X_session, y_session)


def get_cross_session_splits(session_data_dict):
    """
    Protokol B: Cross-session (Leave-One-Session-Out)
    session_data_dict: {"session_001": (X1, y1), "session_002": (X2, y2), ...}
    Her iterasyonda 1 oturumu test, kalanları train yapar.
    Döndürür: List of dicts
    """
    session_names = list(session_data_dict.keys())
    splits = []

    for test_session in session_names:
        train_X_list, train_y_list = [], []
        test_X, test_y = session_data_dict[test_session]

        for train_session in session_names:
            if train_session != test_session:
                train_X_list.append(session_data_dict[train_session][0])
                train_y_list.append(session_data_dict[train_session][1])

        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        splits.append({
            "train_X": train_X, "train_y": train_y,
            "test_X": test_X, "test_y": test_y,
            "test_session_name": test_session
        })

    return splits


def get_multi_session_split(session_data_dict, holdout_session_name, val_session_name):
    """
    Protokol C: Multi-session train + holdout
    Spesifik olarak 1 oturumu validation, 1 oturumu holdout (asla eğitime girmeyen final test) ayırır.
    Kalan tüm oturumları eğitim için birleştirir.
    """
    train_X_list, train_y_list = [], []

    for name, (X, y) in session_data_dict.items():
        if name == holdout_session_name or name == val_session_name:
            continue
        train_X_list.append(X)
        train_y_list.append(y)

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    val_X, val_y = session_data_dict[val_session_name]
    holdout_X, holdout_y = session_data_dict[holdout_session_name]

    return {
        "train": (train_X, train_y),
        "val": (val_X, val_y),
        "holdout": (holdout_X, holdout_y)
    }