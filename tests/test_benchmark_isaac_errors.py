from pathlib import Path

from DynaNav.benchmark import _scan_isaac_log_for_errors


def test_scan_isaac_log_reports_lifecycle_permission_error(tmp_path: Path):
    log_path = tmp_path / "episode.log"
    log_path.write_text(
        "2026-10-29 [Error] [py stderr]: Traceback (most recent call last):\n"
        "2026-10-29 [Error] [py stderr]:   File \"bridge.py\", line 1, in on_play\n"
        "2026-10-29 [Error] [py stderr]: PermissionError: [Errno 13] Permission denied: '/media/disk'\n"
    )

    assert _scan_isaac_log_for_errors(log_path) == [
        "PermissionError: [Errno 13] Permission denied: '/media/disk'"
    ]


def test_scan_isaac_log_ignores_ordinary_isaac_error_labels(tmp_path: Path):
    log_path = tmp_path / "episode.log"
    log_path.write_text(
        "[Error] [omni.kit.app._impl] [py stderr]: COLLISION WARNING: HUMAN\n"
        "[Error] [omni.kit.app._impl] [py stderr]: EPISODE TERMINATED\n"
    )

    assert _scan_isaac_log_for_errors(log_path) == []
