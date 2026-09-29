"""工具契约与真实调用路径；外部依赖用替身，ToolCall/ToolMessage 不模拟。"""

import importlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, ToolException
from pydantic import ValidationError

from agentic_rag.graph import nodes
from agentic_rag.news_retrieval import NewsRetrievalResult
from agentic_rag.research.store import ResearchStore
from agentic_rag.tools import TOOLS, describe_tools, read_news, search_knowledge, search_news, search_web
from agentic_rag.tools.contracts import ToolContext
from agentic_rag.tools.execution import execute_tool

news_module = importlib.import_module('agentic_rag.tools.news')
web_module = importlib.import_module('agentic_rag.tools.web_search')


def test_catalog_schemas_hide_runtime_and_reject_unknown_fields():
    assert {tool.name for tool in TOOLS} == {'search_knowledge', 'search_news', 'search_web', 'read_news'}
    for tool in TOOLS:
        assert isinstance(tool, BaseTool)
        assert tool.response_format == 'content_and_artifact'
        schema = tool.tool_call_schema.model_json_schema()
        assert not {'config', 'tool_context', 'read_version', 'evidence', 'api_key', 'database'} & schema['properties'].keys()
        assert tool.args_schema.model_config['extra'] == 'forbid'
        assert tool.name in describe_tools()
    assert set(read_news.tool_call_schema.model_json_schema()['properties']) == {'evidence_ids', 'goal'}


@pytest.mark.parametrize('arguments', [
    {'query': '  '}, {'query': 'q', 'max_results': 0}, {'query': 'q', 'max_results': 11},
    {'query': 'q', 'max_results': True}, {'query': 'q', 'api_key': 'do-not-log-this-value'},
])
def test_invalid_web_input_stops_before_network_and_does_not_log_raw_extras(arguments):
    events = []
    with patch.object(web_module, 'DDGS') as client:
        with pytest.raises(ValidationError):
            execute_tool(search_web, arguments, context=ToolContext(emit=events.append))
    client.assert_not_called()
    assert events[-1]['phase'] == 'failed'
    assert 'do-not-log-this-value' not in json.dumps(events)


@pytest.mark.parametrize('changes', [
    {'start': '2026-02-30'}, {'start': '2026-10-01', 'end': '2026-09-01'},
    {'start': '20260901'}, {'queries': []}, {'queries': ['a'] * 5}, {'queries': ['', '生猪']},
])
def test_news_scope_validation_also_applies_to_direct_tool_invocation(changes):
    with patch.object(news_module, 'retrieve_news') as retrieve:
        with pytest.raises(ValidationError):
            search_news.invoke({'semantic_query': '研究', 'queries': ['生猪'], **changes})
    retrieve.assert_not_called()


def test_web_returns_real_tool_message_with_full_artifact_and_stable_call_id():
    events = []
    body = '原始网页摘要' * 200
    with patch.object(web_module, 'DDGS') as client:
        client.return_value.text.return_value = [{'body': body, 'title': '公开材料', 'href': 'https://example.test/a'}]
        message = execute_tool(search_web, {'query': '  供需  '}, context=ToolContext(emit=events.append))
    assert isinstance(message, ToolMessage)
    assert message.name == 'search_web'
    assert message.status == 'success'
    assert events[0]['arguments'] == {'query': '供需', 'max_results': 4}
    assert {e['tool_call_id'] for e in events} == {message.tool_call_id}
    assert [e['phase'] for e in events] == ['started', 'finished']
    assert message.artifact.documents[0].page_content == body
    assert message.artifact.documents[0].metadata['content_kind'] == 'search_snippet'
    assert body not in message.content
    assert json.loads(message.content)['count'] == 1
    # artifact 仍可序列化，并非混入一个无法处理的服务对象。
    assert message.model_dump(mode='json')['artifact']['documents'][0]['page_content'] == body


def test_network_failure_is_not_a_successful_empty_search_and_is_not_retried():
    events = []
    with patch.object(web_module, 'DDGS') as client:
        client.return_value.text.side_effect = RuntimeError('network unavailable')
        with pytest.raises(ToolException, match='不是检索零结果'):
            execute_tool(search_web, {'query': '猪价'}, context=ToolContext(emit=events.append))
        assert client.return_value.text.call_count == 1
    assert events[-1]['phase'] == 'failed'
    with patch.object(web_module, 'DDGS') as client:
        client.return_value.text.return_value = []
        empty = execute_tool(search_web, {'query': '不存在的材料'})
    assert empty.artifact.status == 'empty' and empty.status == 'success'


def test_news_preserves_partial_success_dates_and_live_event_identity():
    events = []
    event = {'kind': 'news_page', 'page': 1, 'count': 1, 'has_more': False}

    def retrieve(**kwargs):
        assert kwargs['start'] == kwargs['end'] == ''
        assert kwargs['api_queries'] == ['生猪', '饲料']
        kwargs['on_event'](event)
        return NewsRetrievalResult(items=[{'article_id': 'a', 'title': '生猪报道', 'summary': '价格基准'}],
                                   api_failed=True, api_error='HTTP 503', events=[event])

    with patch.object(news_module, 'retrieve_news', side_effect=retrieve):
        message = execute_tool(search_news, {'semantic_query': '价格预测', 'queries': ['生猪', '饲料']},
                               context=ToolContext(emit=events.append))
    assert message.artifact.status == 'degraded'
    assert message.artifact.documents[0].metadata['content_kind'] == 'news_summary'
    assert message.artifact.warnings == ['HTTP 503']
    assert all(e['tool_call_id'] == message.tool_call_id for e in events)
    assert events[-1]['status'] == 'degraded'


def test_news_total_failure_remains_error_not_no_news():
    with patch.object(news_module, 'retrieve_news', return_value=NewsRetrievalResult(
        items=[], api_failed=True, api_error='HTTP 503')):
        message = execute_tool(search_news, {'semantic_query': '最近新闻', 'queries': ['']})
    assert message.status == 'error' and message.artifact.status == 'error'


def test_knowledge_tool_is_lazy_and_does_not_mutate_retriever_documents():
    document = Document(page_content='分析方法', metadata={'source': 'method.md'})
    with (patch('agentic_rag.ingestion.ensure_index') as ensure,
          patch('agentic_rag.ingestion.get_retriever') as retriever):
        retriever.return_value.invoke.return_value = [document]
        message = execute_tool(search_knowledge, {'query': '供需分析'})
    ensure.assert_called_once_with()
    retriever.return_value.invoke.assert_called_once_with('供需分析')
    assert 'source_type' not in document.metadata
    assert message.artifact.documents[0].metadata['source_type'] == 'vectorstore'


def test_original_read_has_exact_offsets_budget_and_evidence_allowlist():
    original = '能繁母猪' + '长段原文' * 100 + '。\n' + '第一段介绍。\n能繁母猪存栏减少。' * 150
    fetch = Mock(return_value=Document(page_content=original))
    context = ToolContext(read_version=fetch, evidence={
        'E1': Document(page_content='笔记', metadata={'memory_version': 'trusted-version', 'content_kind': 'news_article'})})
    message = execute_tool(read_news, {'evidence_ids': ['E1'], 'goal': '能繁母猪'}, context=context)
    fetch.assert_called_once_with('trusted-version')
    assert sum(len(p['quote']) for p in message.artifact.passages) <= 1600
    for passage in message.artifact.passages:
        assert passage['quote'] == original[passage['start']:passage['end']]
    for passage in json.loads(message.content)['passages']:
        assert passage['quote'] == original[passage['start']:passage['end']]
    assert any(p['excerpted'] for p in json.loads(message.content)['passages'])
    fetch.reset_mock()
    with pytest.raises(ToolException, match='不存在'):
        execute_tool(read_news, {'evidence_ids': ['E1', 'E99'], 'goal': '供给'}, context=context)
    fetch.assert_not_called()


@pytest.mark.parametrize('arguments', [
    {'evidence_ids': ['E1', 'E2', 'E3'], 'goal': '供给'},
    {'evidence_ids': ['../../secret'], 'goal': '供给'},
    {'evidence_ids': ['E1'], 'goal': '供给', 'version_id': 'arbitrary-version'},
])
def test_original_read_rejects_unbounded_or_forged_requests(arguments):
    fetch = Mock()
    with pytest.raises(ValidationError):
        execute_tool(read_news, arguments, context=ToolContext(read_version=fetch))
    fetch.assert_not_called()


def test_missing_original_is_explicit_and_parallel_contexts_do_not_leak():
    with pytest.raises(ToolException, match='证据上下文'):
        execute_tool(read_news, {'evidence_ids': ['E1'], 'goal': '核对'})
    fetch = Mock()
    message = execute_tool(read_news, {'evidence_ids': ['E1'], 'goal': '核对'},
        context=ToolContext(read_version=fetch, evidence={'E1': Document(page_content='网页摘要')}))
    assert message.artifact.status == 'error' and message.artifact.warnings
    fetch.assert_not_called()

    def run(label):
        return execute_tool(read_news, {'evidence_ids': ['E1'], 'goal': label}, context=ToolContext(
            read_version=lambda _version: Document(page_content=label),
            evidence={'E1': Document(page_content='笔记', metadata={'memory_version': label})})).artifact.passages
    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(run, ['甲研究的资料。', '乙研究的资料。']))
    assert left[0]['quote'] == '甲研究的资料。' and right[0]['quote'] == '乙研究的资料。'


def test_real_source_nodes_execute_all_three_tools_and_persist_trace(tmp_path):
    database = ResearchStore(tmp_path / 'research.sqlite')
    events = []
    with (patch('agentic_rag.ingestion.ensure_index'),
          patch('agentic_rag.ingestion.get_retriever') as retriever,
          patch.object(news_module, 'retrieve_news', return_value=NewsRetrievalResult(items=[{'title': '新闻', 'content': '事实'}])),
          patch.object(web_module, 'DDGS') as client,
          patch.object(nodes, 'get_stream_writer', return_value=events.append),
          patch('agentic_rag.research.service.store', return_value=database)):
        retriever.return_value.invoke.return_value = [Document(page_content='方法')]
        client.return_value.text.return_value = [{'body': '公开报道', 'href': 'https://example.test'}]
        result = nodes.collect_sources({'question': '价格预测', 'run_id': 'run', 'reading_recipe': 'test',
            'selected_sources': ['vectorstore', 'news_api', 'web_search'],
            'source_queries': {'vectorstore': '方法', 'news_api': '生猪', 'web_search': '供需'}})
    assert len(result['documents']) == 3
    tool_events = [e for e in events if e['kind'] == 'tool']
    assert len(tool_events) == 6
    assert {e['tool'] for e in tool_events} == {'search_knowledge', 'search_news', 'search_web'}
    assert len({e['tool_call_id'] for e in tool_events}) == 3
    saved = database.events('run')
    assert len(saved) == 6


def test_list_tools_does_not_warm_model_or_open_research_database(monkeypatch):
    from agentic_rag import cli
    monkeypatch.setattr('sys.argv', ['agentic-rag', '--list-tools'])
    output = io.StringIO()
    with (patch.object(cli, 'warm_up_model') as warm,
          patch('agentic_rag.research.service.store') as store,
          redirect_stdout(output)):
        cli.main()
    assert len(json.loads(output.getvalue())) == 4
    warm.assert_not_called()
    store.assert_not_called()


def test_collect_and_supplement_keep_web_failure_visible_and_news_success():
    with (patch.object(news_module, 'retrieve_news', return_value=NewsRetrievalResult(items=[{'title': '有效新闻'}])),
          patch.object(web_module, 'DDGS') as client):
        client.return_value.text.side_effect = RuntimeError('network failure')
        result = nodes.collect_sources({'question': '猪价', 'selected_sources': ['news_api', 'web_search'],
            'source_queries': {'news_api': '生猪', 'web_search': '需求'}})
        assert len(result['documents']) == 1
        assert '不是检索零结果' in result['source_errors']['web_search']
        update = nodes.supplement_sources({'question': '猪价', 'documents': result['documents'],
            'pending_web_queries': ['饲料'], 'retries': 0})
    assert len(update['documents']) == 1
    assert 'web_search' in update['source_errors']


def test_supplement_reuses_the_tool_gateway_with_preserved_dates():
    events = []
    with patch.object(news_module, 'retrieve_news', return_value=NewsRetrievalResult(items=[])) as retrieve:
        nodes.supplement_sources({'question': '预测', 'original_question': '预测', 'documents': [],
            'pending_news_queries': ['供给'], 'retries': 0, '_event_writer': events.append,
            'news_search_plan': {'query': '生猪', 'queries': ['生猪'], 'start': '2026-01-01', 'end': '', 'section': ''}})
    assert retrieve.call_args.kwargs['start'] == '2026-01-01'
    assert retrieve.call_args.kwargs['api_queries'] == ['供给']
    assert [event['phase'] for event in events if event['kind'] == 'tool'] == ['started', 'finished']


def test_terminal_trace_shows_call_id_arguments_failure_and_degradation():
    from agentic_rag.cli import TracePrinter
    output = io.StringIO()
    trace = TracePrinter()
    with redirect_stdout(output):
        trace.show_event({'kind': 'tool', 'tool': 'search_web', 'tool_call_id': 'trace-123',
                          'phase': 'started', 'arguments': {'query': '猪价'}})
        trace.show_event({'kind': 'tool', 'tool': 'search_web', 'phase': 'failed', 'error': '搜索故障', 'error_type': 'ToolException'})
        trace.show_event({'kind': 'tool', 'tool': 'search_news', 'phase': 'finished', 'status': 'degraded',
                          'count': 1, 'elapsed': 0.2, 'warnings': ['只有摘要']})
    text = output.getvalue()
    assert all(value in text for value in ('trace-123', '猪价', 'ToolException', '搜索故障', '降级返回', '只有摘要'))
