import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_anonymity import scan_text_files


def test_text_artifacts_contain_no_identity_metadata():
    assert scan_text_files() == []
