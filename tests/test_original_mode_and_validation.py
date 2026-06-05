from watermarklab.paper_reported import write_paper_reported
from watermarklab.proposal_validation import validate_proposal_params


def test_paper_reported_original_mode_handles_mixed_text_and_numeric_values(tmp_path):
    df = write_paper_reported(tmp_path)
    summary = tmp_path / "paper_reported_summary.csv"
    assert not df.empty
    assert summary.exists()
    assert summary.read_text(encoding="utf-8").startswith("method_id,phase,metric")


def test_default_proposal_validation_contract_matches_current_pywt_mode():
    result = validate_proposal_params()
    assert result["ok"] is True
    assert result["checks"]["dwt_mode"] is True
    assert result["contract"]["dwt_mode"] == "pywt"
