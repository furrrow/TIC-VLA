"""Command telemetry must not interrupt bridge command reception."""
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]

def load_definitions(path, names, namespace):
    """Execute real pure definitions without importing ROS, CUDA or model code."""
    import ast
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(selected) == len(names)
    source = 'from __future__ import annotations\n' + '\n'.join((ast.unparse(node) for node in selected))
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace

def test_bridge_trace_receives_clipped_command_and_cannot_interrupt_control(monkeypatch):
    import os
    import json
    import numpy as np
    path = ROOT / 'DynaNav/behavior/nova_carter_ros_bridge.py'
    records = []
    logger = SimpleNamespace(log_info=records.append)
    ns = load_definitions(path, {'NovaCarterROSBridge'}, dict(BehaviorScript=object, os=os, json=json, np=np, carb=logger, time=SimpleNamespace(monotonic=lambda : 100.0, monotonic_ns=lambda : 100000000000)))
    node = ns['NovaCarterROSBridge'].__new__(ns['NovaCarterROSBridge'])
    (node._max_linear_speed, node._max_angular_speed) = (1.5, 1.2)
    (node._episode_id, node._last_sim_time) = ('test', 1.0)
    msg = SimpleNamespace(linear=SimpleNamespace(x=2.0), angular=SimpleNamespace(z=0.0))
    monkeypatch.setenv('TICVLA_COMMAND_TRACE', 'true')
    node._on_command(msg)
    assert node._command == (1.5, 0.0)
    assert json.loads(records[0].split('COMMAND_TRACE ', 1)[1])['receive_sequence'] == 1

    def fail(_):
        raise RuntimeError('diagnostic writer unavailable')
    logger.log_info = fail
    node._on_command(msg)
    assert node._command_time == 100.0
    assert node._command_sequence == 2
