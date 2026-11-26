"""Control-flow tests with explicitly labeled research/model doubles."""
from pathlib import Path

import pytest

from loop_research.engine import Research
from loop_research.models import Plan, Project, SourceRejection
from test_workflow import RadarDouble, WorkerDouble, prepare_research


@pytest.fixture
def engine(tmp_path):
    project = Project.model_validate_json((Path(__file__).parents[1]/'examples/breast-cancer.json').read_bytes())
    project.max_experiments = 6
    result = Research.initialize(tmp_path, project)
    result.worker = WorkerDouble()
    result.radar = RadarDouble(result.evidence)
    return result


def test_initial_research_precedes_every_executable_hypothesis(engine):
    assert engine.tick().phase == 'research'
    assert not engine.worker.calls
    assert engine.tick().research_stage == 'fetch'
    assert engine.tick().research_stage == 'screen'
    assert engine.tick().phase == 'plan'
    state = engine.store.read()
    pool = engine.evidence.get(state.direction_pool)['content']
    assert len(pool['directions']) == 3 and pool['mode'] == 'initial'
    assert len(engine.radar.history()) == 2
    state = engine.tick()
    assert state.phase == 'execute'
    directions = {d['id']:d for d in pool['directions']}
    for task in state.active_tasks:
        selected = directions[task.hypothesis.direction_id]
        assert task.hypothesis.machine_target == selected['machine_target']
        assert {q['source_id'] for q in selected['quotes']} <= set(task.hypothesis.source_ids)
    assert not list(engine.platform.root.glob('*/request.json'))


def test_planner_knowledge_gap_routes_to_progressive_research_before_training(engine):
    engine = prepare_research(engine)
    original = engine.worker
    class Gap:
        def run(self, role, payload, response, **kwargs):
            if role == 'planner':
                # Compact directions carry quotes/IDs; full papers are not repeatedly injected.
                assert 'body_excerpt' not in str(payload['direction_pool'])
                return Plan(rationale='TEST new information gap', hypotheses=[], research_questions=['TEST calibration evidence gap']), {'TEST_DOUBLE': True}
            return original.run(role, payload, response, **kwargs)
    engine.worker = Gap()
    state = engine.tick()
    assert state.phase == 'research' and state.research_mode == 'progressive'
    assert state.research_questions == ['TEST calibration evidence gap']
    assert state.used == 0 and not list(engine.platform.root.glob('*/request.json'))


def test_repeated_queries_are_rejected_without_burning_training_budget(engine):
    engine.tick()
    engine.tick()
    request = engine.evidence.get(engine.store.read().research_request)['content']['request']
    engine.radar.queries = [q['query'] for q in request['queries']]
    old = engine.worker
    class Repeated:
        def run(self, role, payload, response, **kwargs):
            payload = {**payload, 'previous_queries': []}
            return old.run(role,payload,response,**kwargs)
    engine.worker = Repeated()
    engine.store.change(lambda s:setattr(s,'research_stage','query'),event='TEST replay query')
    for _ in range(3):
        state = engine.tick()
    assert state.phase == 'review' and state.research_failures == 3
    assert state.used == 0 and not state.active_tasks


def test_stagnation_forces_research_and_excludes_previous_dimensions(engine):
    engine = prepare_research(engine)
    engine.tick()
    baseline = engine.evidence.get(engine.store.read().baseline_evidence)['content']
    code = engine.evidence.get(engine.store.read().best_code)['content']['code']
    model = engine.evidence.put('model_code', {'code':code,'diff':'TEST FIXTURE no-op candidate','commit':'TEST'}, [engine.store.read().best_code])
    measured = engine.evidence.put('training_result',baseline,[model])
    def synthetic_batch(state):
        state.used, state.phase = 3, 'converge'
        for task in state.active_tasks:
            task.status,task.counted = 'succeeded',True
            task.metrics,task.code_artifact,task.evidence = baseline['metrics'],model,measured
    engine.store.change(synthetic_batch,event='TEST synthetic no-gain batch')
    state = engine.tick()
    previous = set(state.research_avoid_targets)
    assert state.phase == 'curate' and state.after_curation == 'research'
    assert len(previous) == 3 and len(state.attempted_directions) == 3
    assert engine.tick().phase == 'research'
    for _ in range(3):
        state = engine.tick()
    assert state.phase == 'plan' and state.research_mode == 'escape'
    directions = engine.evidence.get(state.direction_pool)['content']['directions']
    assert all(d['machine_target'] not in previous for d in directions)
    assert len(engine.radar.history()) == 4
    assert not list(engine.platform.root.glob('*/request.json'))


def test_partial_valid_directions_survive_while_only_missing_targets_are_researched(engine):
    original = engine.worker
    class PartialThenComplete:
        screens = 0
        def run(self, role, payload, response, **kwargs):
            if role == 'research-query' and self.screens:
                assert set(payload['covered_targets']) == {'preprocessing','classifier.LogisticRegression.C'}
            result,meta = original.run(role,payload,response,**kwargs)
            if role == 'literature-screen':
                self.screens += 1
                if self.screens == 1:
                    result.directions = result.directions[:2]
                else:
                    result.directions = result.directions[2:]
                used = {quote.source_id for d in result.directions for quote in d.quotes}
                result.excluded_sources = [SourceRejection(source_id=key,reason='TEST not needed for this gap') for key in payload['source_ids'] if key not in used]
            return result,meta
    engine.worker = PartialThenComplete()
    for _ in range(4):
        state = engine.tick()
    first = state.direction_pool
    assert state.phase == 'research' and state.research_stage == 'query'
    pool = engine.evidence.get(first)['content']
    assert len(pool['directions']) == 2 and not pool['ready']
    for _ in range(3):
        state = engine.tick()
    complete = engine.evidence.get(state.direction_pool)
    assert state.phase == 'plan' and complete['content']['ready']
    assert {d['id'] for d in pool['directions']} <= {d['id'] for d in complete['content']['directions']}
    assert first in complete['parents'] and state.used == 0


def test_new_best_code_requires_reassessing_old_direction_pool(engine):
    engine = prepare_research(engine)
    current = engine.store.read()
    code = engine.evidence.get(current.best_code)['content']['code']
    updated = engine.evidence.put('model_code', {'code':code.replace('C=1.0','C=0.1'), 'diff':'TEST simulated best-version change','commit':'TEST'}, [current.best_code])
    engine.store.change(lambda s:setattr(s,'best_code',updated),event='TEST simulate measured-best version update')
    state = engine.tick()
    assert state.phase == 'research' and state.research_mode == 'progressive'
    assert engine.worker.calls == []
