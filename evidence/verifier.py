"""Root-level export forwarding to satquery.backend.evidence.verifier."""

import os
import sys

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from satquery.backend.evidence.verifier import (
    EvidenceBundle,
    EvidenceVerifier,
    evidence_verifier,
    bundle_evidence,
    combine_evidence,
    verify_and_bundle,
)

__all__ = [
    "EvidenceBundle",
    "EvidenceVerifier",
    "evidence_verifier",
    "bundle_evidence",
    "combine_evidence",
    "verify_and_bundle",
]
