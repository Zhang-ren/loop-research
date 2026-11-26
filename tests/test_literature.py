"""Source checks use labeled fixture text; live retrieval is reported separately."""
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from loop_research.literature import LiteratureRadar, public_url, validate_screen
from loop_research.models import DirectionProposal, LiteratureScreen, PaperQuote, SourceRejection
from loop_research.storage import Evidence


@pytest.fixture
def sources(tmp_path):
    evidence = Evidence(tmp_path)
    keys = [evidence.put('paper_source', {'title': f'TEST paper {n}', 'retracted': False,
        'access': 'body_excerpt', 'abstract': 'TEST supplied abstract about regularized classification.',
        'body_excerpt': f'TEST ORIGINAL observation {n}: feature scaling changes the optimization geometry.'}) for n in range(2)]
    return evidence, keys


def screen(keys, quote='TEST ORIGINAL observation 0: feature scaling changes the optimization geometry.'):
    direction = DirectionProposal(title='TEST scale hypothesis', machine_target='preprocessing',
        hypothesis='TEST: scaling may improve the fixed benchmark metrics.', rationale='TEST supplied paper supports this intervention.',
        parameter_bounds='TEST fixed transform', quotes=[PaperQuote(source_id=keys[0], field='body', quote=quote)],
        priority=90, limitations=['TEST text is not actual literature'])
    return LiteratureScreen(summary='TEST screen', directions=[direction],
        excluded_sources=[SourceRejection(source_id=keys[1], reason='TEST not applicable')], unresolved=[])


def test_quote_is_anchored_to_original_source_and_every_source_is_triaged(sources):
    evidence, keys = sources
    result = validate_screen(screen(keys), keys, ['preprocessing'], evidence)
    assert len(result) == 1 and result[0]['trust'] == 'screened_hypothesis_not_validated_fact'
    with pytest.raises(ValueError, match='quotation not present'):
        validate_screen(screen(keys, 'This fabricated observation does not occur in the supplied original.'), keys, ['preprocessing'], evidence)
    incomplete = screen(keys)
    incomplete.excluded_sources = []
    with pytest.raises(ValueError, match='explicit use or exclusion'):
        validate_screen(incomplete, keys, ['preprocessing'], evidence)


@pytest.mark.parametrize('access,retracted', [('metadata_only', False), ('body_excerpt', True)])
def test_inaccessible_or_retracted_papers_cannot_authorize_directions(tmp_path, access, retracted):
    evidence = Evidence(tmp_path)
    value = {'title': 'TEST source', 'access': access, 'retracted': retracted,
             'abstract': 'TEST abstract', 'body_excerpt': 'TEST ORIGINAL observation 0: feature scaling changes the optimization geometry.'}
    a = evidence.put('paper_source', value)
    b = evidence.put('paper_source', {**value, 'title': 'TEST excluded'})
    with pytest.raises(ValueError):
        validate_screen(screen([a,b]), [a,b], ['preprocessing'], evidence)


def test_source_identity_and_allowed_intervention_are_not_model_decisions(sources):
    evidence, keys = sources
    with pytest.raises(ValueError, match='permitted intervention'):
        validate_screen(screen(keys), keys, ['classifier.family'], evidence)
    item = screen(keys)
    item.directions[0].quotes[0].source_id = 'f'*64
    with pytest.raises(ValueError, match='Invented'):
        validate_screen(item, keys, ['preprocessing'], evidence)
    item = screen(keys)
    item.directions[0].machine_target = 'classifier.family'
    item.directions[0].classifier_name = 'LinearDiscriminantAnalysis'
    with pytest.raises(ValueError, match='available concrete classifier'):
        validate_screen(item, keys, ['classifier.family'], evidence, classifier_names={'SVC','RandomForestClassifier'})


def test_search_cache_reuses_raw_result_without_repeating_network(tmp_path, monkeypatch):
    radar = LiteratureRadar(tmp_path, Evidence(tmp_path))
    calls = []
    def run(command, **kwargs):
        calls.append(json.loads(kwargs['input']))
        return subprocess.CompletedProcess(command, 0, json.dumps({'records': []}).encode(), b'')
    monkeypatch.setattr(subprocess, 'run', run)
    first = radar._search('TEST   Scaling')
    second = radar._search('test scaling')
    assert first['evidence'] == second['evidence'] and second['cache_hit']
    assert len(calls) == 1 and radar.history() == ['TEST   Scaling']


def test_read_cache_reuses_one_version_of_an_accessible_paper(tmp_path, monkeypatch):
    evidence = Evidence(tmp_path)
    radar = LiteratureRadar(tmp_path, evidence)
    search_id = evidence.put('paper_search', {'TEST_DOUBLE': True})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        folder = Path(command[command.index('--papers-dir')+1])
        path = folder/'TEST-paper.md'
        path.write_text('TEST Logistic Regression Feature Scaling. ' * 100)
        return subprocess.CompletedProcess(command, 0, json.dumps({'full_text_path':str(path)}).encode(), b'')
    monkeypatch.setattr(subprocess, 'run', run)
    record = {'id':'https://openalex.org/TEST', 'doi':'https://doi.org/TEST-scaling', 'title':'TEST Logistic Regression Feature Scaling',
              'best_oa_location':{'pdf_url':'https://example.org/TEST.pdf'}, 'abstract_text':'TEST abstract'}
    first = radar._read(record, search_id)
    second = radar._read({**record,'doi':'https://doi.org/test-SCALING'}, search_id)
    assert first == second and len(calls) == 1
    paper = evidence.get(first)['content']
    assert paper['access'] == 'body_excerpt' and paper['body_evidence']
    evidence.verify_tree(first)


@pytest.mark.parametrize('url', ['http://example.org/a','https://127.0.0.1/a','https://10.0.0.2/a','https://localhost/a','https://x.local/a','https://user:secret@example.org/a'])
def test_paper_reader_rejects_local_or_credential_bearing_urls(url):
    assert not public_url(url)


def test_failed_source_can_use_new_open_location_without_erasing_its_previous_version(tmp_path, monkeypatch):
    evidence = Evidence(tmp_path)
    radar = LiteratureRadar(tmp_path, evidence)
    search = evidence.put('paper_search', {'TEST_DOUBLE': True})
    calls = []
    def run(command, **kwargs):
        url = command[command.index('--url')+1]
        calls.append(url)
        if url == 'https://example.org/unavailable.pdf':
            return subprocess.CompletedProcess(command, 1, b'', b'TEST HTTP 403')
        folder = Path(command[command.index('--papers-dir')+1])
        target = folder/'TEST-open-paper.md'
        target.write_text('TEST Logistic Regression Feature Scaling. ' * 100)
        return subprocess.CompletedProcess(command, 0, json.dumps({'full_text_path':str(target)}).encode(), b'')
    monkeypatch.setattr(subprocess, 'run', run)
    record = {'id':'https://openalex.org/TEST', 'doi':'https://doi.org/TEST', 'title':'TEST Logistic Regression Feature Scaling',
              'best_oa_location':{'pdf_url':'https://example.org/unavailable.pdf'}}
    # The DOI itself is a candidate too; keep this test's initial locations explicit.
    record['doi'] = None
    first = radar._read(record, search)
    assert evidence.get(first)['content']['access'] == 'metadata_only'
    second = radar._read({**record, 'locations':[{'is_oa':True,'pdf_url':'https://example.net/open.pdf'}]},search)
    assert evidence.get(second)['content']['access'] == 'body_excerpt'
    assert first in evidence.get(second)['parents']
    assert evidence.get(first)['content']['access'] == 'metadata_only'
    assert evidence.get(second)['content']['url'] == 'https://example.net/open.pdf'
    assert calls.count('https://example.org/unavailable.pdf') == 1


def test_arxiv_pdf_falls_back_to_verified_abstract_page_without_indexed_abstract(tmp_path, monkeypatch):
    evidence = Evidence(tmp_path)
    radar = LiteratureRadar(tmp_path, evidence)
    search = evidence.put('paper_search', {'TEST_DOUBLE': True})
    abstract = 'TEST paper abstract about feature scaling and observing how a deterministic baseline changes. ' * 2
    hits = []

    def handler(request):
        hits.append(str(request.url))
        return httpx.Response(200, text=f'<h1>TEST Feature Scaling Experiment</h1>'
                                       f'<blockquote class="abstract">Abstract: {abstract}</blockquote>')

    client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, b'', b'TEST HTTP 403'))
    record = {'id': 'https://openalex.org/TEST', 'title': 'TEST Feature Scaling Experiment',
              'best_oa_location': {'pdf_url': 'https://arxiv.org/pdf/2202.07925'}}
    first = radar._read(record, search)
    paper = evidence.get(first)['content']
    assert paper['access'] == 'abstract_page' and paper['abstract'].strip() == abstract.strip()
    assert paper['landing_page_sha256'] and paper['body_evidence'] is None
    assert radar._read(record, search) == first
    assert hits == ['https://arxiv.org/abs/2202.07925']
