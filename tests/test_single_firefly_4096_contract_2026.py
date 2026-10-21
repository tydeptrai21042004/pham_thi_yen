"""Short, deterministic tests for single-firefly/4096-only run configuration."""
from __future__ import annotations
import ast
import hashlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from watermarklab.methods.proposal_qh_dwt_hess import ProposalQHDWTHess


def test_one_firefly_is_accepted_and_does_not_generate_fake_movements(monkeypatch):
    model = ProposalQHDWTHess(optimizer_trials=1, optimizer_seed=3)
    calls = []
    def lightweight_objective(host, wm, params, attack_suite=None):
        calls.append(params.to_dict())
        return {"objective": 0.5, "clean_psnr": 55.1, "mean_attack_nc": 0.8}
    monkeypatch.setattr(model, '_evaluate_params_for_optimization', lightweight_objective)
    _, record = model.optimize_params(np.zeros((8,8,3),dtype=np.uint8),
                  np.zeros((4,4),dtype=np.uint8), n_fireflies=1,
                  n_generations=2, seed=99, attack_suite=[])
    assert record['optimizer']['n_fireflies'] == 1
    assert record['optimizer']['n_generations'] == 2
    assert len(record['history']) == 1  # one candidate, no peer interaction
    assert len(calls) == 2  # initial and final verification of same params
    assert calls[0] == calls[1]


def test_launcher_enables_only_4096_neural_extensions():
    source=(ROOT/'KAGGLE_GITHUB_ALL_EXPERIMENTS.py').read_text()
    tree=ast.parse(source)
    assigned={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if isinstance(node.value,ast.Constant):
                assigned[node.targets[0].id]=node.value.value
    assert assigned['FIREFLIES']==1
    assert assigned['GENERATIONS']==2
    assert assigned['RUN_4096_EXTENSIONS'] is True
    assert assigned['RUN_ALL_SENSITIVITY_HOSTS'] is True
    assert '--include-neural4096' in source
    assert "'--step', 'all'" in source
    assert "'--phase', 'neural'" not in source


def test_optimizer_source_remains_unchanged():
    proposal=ROOT/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
    assert 'h_selection_policy: str = "manuscript"' in proposal.read_text(encoding='utf-8')
