"""Verify the current batch, source/implementation chain and actual reproducibility."""
from pathlib import Path
import argparse
import json
import subprocess
import tempfile

from mlflow import MlflowClient

from loop_research.command_candidate import STRUCTURE_TARGET, candidate_parts, structural
from loop_research.engine import Research, actual_classifier, classifier_names
from loop_research.literature import validate_screen
from loop_research.models import DirectionProposal, LiteratureScreen
from loop_research.model_policy import LIMITS, validate_proposal
from loop_research.replay import reproduce
from loop_research.storage import atomic, canonical, digest


def verify(root: Path, replay=False):
    engine = Research(root)
    state = engine.store.read()
    engine.references()
    assert state.phase == 'complete' and state.final_evidence
    for path in engine.evidence.root.glob('*.json'):
        engine.evidence.verify_tree(path.stem)
    pool_record = engine.evidence.get(state.direction_pool) if state.direction_pool else None
    pool = pool_record['content'] if pool_record else None
    pool_checks = {}
    if pool:
        allowed = engine.targets(engine.evidence.get(pool['baseline_code'])['content']['code'])
        if engine.command and structural(engine.command):
            retrieval = engine.evidence.get(pool_record['parents'][0])
            request = engine.evidence.get(retrieval['parents'][0])
            assert retrieval['kind'] == 'literature_retrieval' and request['kind'] == 'research_request'
            allowed = request['content']['allowed_targets']
        sources = sorted({q['source_id'] for d in pool['directions'] for q in d['quotes']})
        models = [DirectionProposal.model_validate({k:v for k,v in d.items() if k in DirectionProposal.model_fields}) for d in pool['directions']]
        checked = validate_screen(LiteratureScreen(summary='Controller provenance audit', directions=models,
                    excluded_sources=[], unresolved=[]), sources, allowed, engine.evidence, classifier_names=classifier_names())
        assert {d['id'] for d in checked} == {d['id'] for d in pool['directions']}
        directions = {d['id']:d for d in checked}
        for task in state.active_tasks:
            if task.mode != 'candidate':
                continue
            expected = directions[task.hypothesis.direction_id]
            code = engine.evidence.get(task.code_artifact)['content']['code']
            before = engine.evidence.get(engine.evidence.get(task.code_artifact)['parents'][0])['content']['code']
            assert engine.intervention(before,code) == expected['machine_target']
            assert task.hypothesis.classifier_name == expected['classifier_name']
            if expected['machine_target'] == STRUCTURE_TARGET:
                artifact = engine.evidence.get(task.code_artifact)
                review = engine.evidence.get(artifact['content']['structural_review'])
                assert review['kind'] == 'structural_candidate_review'
                assert review['content']['review']['accepted']
                assert review['content']['candidate_sha256'] == digest(code.encode())
            if expected['classifier_name']:
                assert actual_classifier(code) == expected['classifier_name']
        pool_checks = {'valid_directions':len(checked), 'quoted_sources':len(sources), 'executed_directions':len(state.active_tasks),
                       'all_quotes_match_raw_text':True, 'implementations_match_paper_supported_hypotheses':True}
    analysis_record=engine.evidence.get(state.last_analysis)
    cited=set(analysis_record['content']['analysis']['evidence_ids'])
    assert cited <= set(analysis_record['parents'])
    assert {t.evidence for t in state.active_tasks if t.evidence} <= cited
    metadata=analysis_record['content']['worker']
    wire_schema=json.loads((Path(metadata['directory'])/'schema.json').read_bytes())
    wire_ids=wire_schema['properties']['evidence_ids']['items'].get('enum')
    if wire_ids:
        assert cited <= set(wire_ids)
    client = MlflowClient(tracking_uri='sqlite:///'+str(engine.root/'mlflow.db'))
    experiment = client.get_experiment_by_name(engine.project.name)
    runs=[]
    filename = 'model.py'
    if engine.command:
        filename = 'candidate.json' if structural(engine.command) else 'parameters.json'
    with tempfile.TemporaryDirectory(prefix='loop-mlflow-audit-') as folder:
        for run in client.search_runs([experiment.experiment_id]):
            key=run.data.tags['evidence_id']
            result=engine.evidence.get(key)['content']
            code=Path(client.download_artifacts(run.info.run_id,filename,folder)).read_bytes()
            assert run.data.metrics == result['metrics']
            assert digest(code) == result['code_sha256']
            if engine.command:
                candidate_parts(code.decode(), engine.command)
            else:
                validate_proposal(code.decode(), engine.project.seed)
            runs.append({'run_id':run.info.run_id,'evidence':key,'metrics_match':True,'code_matches':True})
    for task in state.active_tasks:
        original=engine.evidence.get(task.code_artifact)['content']['code']
        code=subprocess.check_output(['git','-C',str(engine.root/'candidates'/task.id),'show',task.code_commit+':'+filename]).decode()
        assert code == original
    holdout=engine.evidence.get(state.final_evidence)['content']
    selected=engine.evidence.get(state.best_evidence)['content']
    assert holdout['split']=='test' and selected['split']=='validation'
    assert not set(holdout['evaluation_indices']) & (set(selected['train_indices']) | set(selected['evaluation_indices']))
    assert state.offline_accepted == all(m.passes(holdout['metrics'][m.name]) for m in engine.project.metrics)
    replays=[]
    if replay:
        for key in [state.best_evidence,state.final_evidence]:
            record=reproduce(engine,key)
            assert record['passed']
            replays.append(record['evidence'])
        atomic(engine.root/'reproduction-ref.json',canonical({'evidence_ids':replays}))
    elif (engine.root/'reproduction-ref.json').exists():
        replays=json.loads((engine.root/'reproduction-ref.json').read_bytes())['evidence_ids']
        for key in replays:
            engine.evidence.verify_tree(key)
            reproduction=engine.evidence.get(key)
            assert reproduction['kind']=='reproduction' and reproduction['content']['passed']
            assert reproduction['content']['original'] in {state.best_evidence,state.final_evidence}
    result={'passed':True,'scope':'current complete batch and frozen selected-candidate holdout; not all 82 acceptances',
            'pool':pool_checks,'mlflow_runs':runs,'candidate_git_commits_verified':len(state.active_tasks),
            'analysis_cites_every_branch':True,'wire_evidence_ids_bound':bool(wire_ids),
            'holdout_disjoint':True,'offline_accepted':state.offline_accepted,'reproductions':replays,
            'formal_memory_count':len(engine.memory.search('',state.project_id))}
    if engine.command:
        source = engine.platform.source
        assert digest((engine.root/'source.tar').read_bytes()) == source['snapshot_sha256']
        assert holdout['evaluation_indices'] == source['splits']['test']
        reviews = []
        for path in sorted((engine.root/'reviews').glob('*-replay.json')):
            review_key = json.loads(path.read_bytes())['evidence']
            replay_record = engine.evidence.get(review_key)
            assert replay_record['kind'] == 'independent_review_replay'
            reviews.append({'evidence': review_key, 'matched': replay_record['content']['matched']})
        result['command_experiment_check'] = {'git_revision': source['revision'], 'snapshot_matches': True,
            'allowed_parameters_checked': len(runs), 'independent_replays': reviews}
    else:
        import loop_research.model_policy as policy
        result['submission_policy_check'] = {'passed':True, 'training_artifacts_checked':len(runs),
            'limits':LIMITS, 'policy_sha256':digest(Path(policy.__file__).read_bytes()),
            'scope':'Compatibility of recorded factories; no training repeated and no boundary probes claimed'}
    key=engine.evidence.put('run_layer_verification',result,[state.direction_pool,state.best_evidence,state.final_evidence] if pool else [state.best_evidence,state.final_evidence])
    atomic(engine.root/'layer-verification-ref.json',canonical({'evidence':key}))
    engine.report()
    return {**result,'evidence':key}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.replay),ensure_ascii=False,indent=2))
