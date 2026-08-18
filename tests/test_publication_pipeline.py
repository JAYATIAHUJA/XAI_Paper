import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cefhf_architecture"))

import common
import publication_experiments as pe


def test_repository_data_path_resolves():
    assert Path(common.XAI_PAPER, "data", "adult", "adult.data").exists()


def test_reweighing_equalizes_joint_mass():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    a = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    w = pe.reweigh(y, a)
    assert np.isclose(w.mean(), 1)
    for g in np.unique(a):
        for label in np.unique(y):
            assert np.isclose(w[(a == g) & (y == label)].sum(), 2.0)


def test_group_metrics_are_bounded():
    m = pe.group_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]),
                         np.array(["p", "p", "u", "u"]), "p")
    assert 0 <= m["SPD"] <= 1
    assert 0 <= m["DIR"] <= 1
    assert 0 <= m["EOD"] <= 1


def test_preprocessor_handles_unseen_category():
    train = pd.DataFrame({"x": [1, 2], "c": ["a", "b"]})
    test = pd.DataFrame({"x": [3], "c": ["new"]})
    pre = pe.make_preprocessor(train)
    pre.fit(train)
    assert pre.transform(test).shape[0] == 1
