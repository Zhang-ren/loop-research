from loop_research.models import Plan, Analysis
from loop_research.workers import response_schema


def test_nullable_backward_compatible_field_is_still_required_in_wire_schema():
    schema = response_schema(Plan)
    hypothesis = schema["$defs"]["Hypothesis"]
    # Regression for a real Codex service 400, rather than a planner reasoning error.
    assert "machine_target" in hypothesis["required"]
    assert hypothesis["properties"]["machine_target"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert "default" not in hypothesis["properties"]["machine_target"]


def test_wire_schema_only_allows_actual_evidence_and_direction_identities():
    ids=['a'*64,'b'*64]
    schema=response_schema(Analysis,evidence_ids=ids)
    assert schema['properties']['evidence_ids']['items']['enum'] == ids
    schema=response_schema(Plan,source_ids=ids,direction_ids=['d'*64],machine_targets=['preprocessing'])
    hypothesis=schema['$defs']['Hypothesis']['properties']
    assert hypothesis['source_ids']['items']['enum'] == ids
    assert hypothesis['direction_id']['anyOf'][0]['enum'] == ['d'*64]
