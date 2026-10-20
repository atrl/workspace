import pytest

from src.factor.research_candidates import (CandidateSpec, correlation_report, expression_digest,
    failure_record, memo_key, reopening_assessment, validate_batch)


def _candidate(**changes):
    values = dict(candidate_id="factor-a", mechanism="path response", receiver="allocation",
        expression="Mean($close, 5) / Ref($close, 1) - 1", allowed_fields=("close",),
        fields_identity="fields-sha", available_at_rule="next session 09:31",
        target_rule="SOXL minus SOXX next open-to-open", maturity_rule="five sessions plus one",
        fixed_parameters={"window": 5}, matched_control="same model without this input",
        rejection_rule="increment absent or net account fails", source_urls=("https://example.org/paper",),
        prior_route="closed route A", new_evidence_explanation="new source vintage", parent_route_closed=True)
    values.update(changes)
    return CandidateSpec(**values)


def test_candidate_funnel_security_duplicates_memo_and_closed_route():
    spec = _candidate()
    audited = validate_batch([spec])
    assert audited["factor-a"]["expression_sha256"] == expression_digest(spec)
    assert validate_batch([_candidate(candidate_id="renamed")])
    with pytest.raises(ValueError, match="duplicate canonical"):
        validate_batch([spec, _candidate(candidate_id="renamed")])
    with pytest.raises(ValueError, match="1..8"):
        validate_batch([_candidate(candidate_id=f"c{i}", expression=f"Mean($close, 5) + {i}")
                        for i in range(9)])
    with pytest.raises(ValueError, match="two mechanisms"):
        validate_batch([_candidate(candidate_id=f"m{i}", mechanism=f"mechanism-{i}",
                                   expression=f"Mean($close, {i + 1})") for i in range(3)])
    with pytest.raises(ValueError, match="one complete economic"):
        validate_batch([_candidate(candidate_id=f"e{i}", expression=f"Mean($close, {i + 1})",
                                   complete_economic_experiment=True) for i in range(2)])
    for unsafe in ("__import__('os').system('id')", "Ref($close, -1)",
                   "Mean($close, $window)", "Mean($close, 0)", "Mean($secret, 5)",
                   "__field_secret + $close", "$close + Mean", "$close > 0 and $close < 2",
                   "not $close", "Quantile($close, 5)", "Quantile($close, 5, 1.1)"):
        with pytest.raises(ValueError):
            expression_digest(_candidate(expression=unsafe))
    assert expression_digest(_candidate(expression="Quantile($close, 5, 0.75)"))

    key = memo_key(spec, data_identity="data-1", code_identity="code-1", split_identity="dev:cutoff-1")
    prose_only = _candidate(expression=" Mean ( $close , 5 ) / Ref ( $close , 1 ) - 1 ",
        source_urls=("https://other.example/source",), prior_route="new wording",
        new_evidence_explanation="edited prose")
    assert memo_key(prose_only, data_identity="data-1", code_identity="code-1",
                    split_identity="dev:cutoff-1") == key
    assert memo_key(spec, data_identity="data-2", code_identity="code-1",
                    split_identity="dev:cutoff-1") != key
    assert memo_key(spec, data_identity="data-1", code_identity="code-2",
                    split_identity="dev:cutoff-1") != key
    assert key != memo_key(spec, data_identity="data-1", code_identity="code-1", split_identity="dev:cutoff-2")
    code = _candidate(expression_kind="code_identity",
        expression="src/workflow/p2.py#" + "a" * 64, allowed_fields=(), code_review_ref="review.json")
    assert validate_batch([code])["factor-a"]["code_review_status"] == "needs_code_review"

    failure = failure_record(spec, failed_gate="increment", reason="no validated gain")
    denied = reopening_assessment(failure, changed_parameters=("window",))
    assert not denied["eligible_for_independent_review"]
    assert denied["parameter_only_rescue_rejected"]
    reviewed = reopening_assessment(failure, reviewed_mechanism_change="review-2")
    assert reviewed["eligible_for_independent_review"] and reviewed["review_required"]
    assert not reviewed["route_reopened"]
    new_route_failure = failure_record(_candidate(parent_route_closed=False),
        failed_gate="increment", reason="first version failed")
    assert new_route_failure["route_state"] == "closed" and new_route_failure["parent_route_state"] == "open"
    assert not reopening_assessment(new_route_failure, changed_parameters=("window",))["eligible_for_independent_review"]

    corr = correlation_report({"a": [1, 2, 3], "b": [2, 4, 6]}, scope="development")
    assert corr["correlation"]["a"]["b"] == pytest.approx(1.)
    assert corr["selected_candidate"] is None and not corr["orthogonalized"]
    with pytest.raises(ValueError, match="development scope"):
        correlation_report({"a": [1, 2], "b": [2, 3]}, scope="outer")
