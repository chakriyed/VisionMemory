import json
from unittest.mock import patch, MagicMock

import pytest

from vision_memory.scene import scene_facts, load_scene
from vision_memory.chat import SceneChat


def scene(*items, scale=1):
    return {'width': 1000 * scale, 'height': 1000 * scale,
            'detections': [{'label': label, 'bbox': [v * scale for v in box], 'confidence': .9}
                           for label, box in items]}


def test_next_to_and_direction_and_overlap():
    facts = scene_facts(scene(('wallet', (100, 100, 200, 200)),
                              ('pen', (220, 100, 250, 200)),
                              ('table', (0, 0, 900, 900))))
    assert facts['spatial_relations'][0]['relations'] == ['left of', 'next to']
    assert facts['spatial_relations'][1]['relations'] == ['boxes overlap']
    assert facts['next_to']['wallet #1'] == ['pen #1']
    assert facts['next_to']['pen #1'] == ['wallet #1']
    assert facts['next_to']['table #1'] == []
    assert not facts['comparison']['available']


def test_movement_requires_camera_assertion():
    before = scene(('wallet', (100, 100, 200, 200)))
    after = scene(('wallet', (400, 100, 500, 200)))
    assert 'camera motion unknown' in scene_facts(after, before)['comparison']['changes'][0]['status']
    assert scene_facts(after, before, True)['comparison']['changes'][0]['status'] == 'possibly moved'
    assert scene_facts(before, scene(('wallet', (100, 100, 200, 200)), scale=2), True)['comparison']['changes'][0]['status'] == 'no significant center change'


def test_multiple_instances_are_ambiguous_not_falsely_matched():
    before = scene(('pen', (0, 0, 10, 10)), ('pen', (40, 40, 50, 50)))
    after = scene(('pen', (300, 300, 310, 310)), ('wallet', (20, 20, 40, 40)))
    changes = scene_facts(after, before, True)['comparison']['changes']
    assert changes[0]['status'] == 'ambiguous identity: multiple instances'
    assert changes[1]['status'] == 'newly detected'
    assert 'may be missed' in scene_facts(scene(), before)['comparison']['changes'][0]['status']


def test_scene_validation(tmp_path):
    path = tmp_path / 'scene.json'
    path.write_text(json.dumps(scene(('pen', (0, 0, 10, 10)))))
    assert len(load_scene(path)['detections']) == 1
    path.write_text(json.dumps(scene(('pen', (30, 0, 10, 10)))))
    with pytest.raises(ValueError, match='Cannot read scene'):
        load_scene(path)


def test_chat_sends_facts_and_remembers_followups():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({'message': {'content': 'The pen is next to wallet #1.'}}).encode()
    chat = SceneChat({'objects': []}, 'test-model')
    with patch('vision_memory.chat.urlopen', return_value=response) as post:
        assert 'pen' in chat.ask('What is next to the wallet?')
        chat.ask('And above it?')
    request = json.loads(post.call_args.args[0].data)
    assert request['stream'] is False
    assert request['model'] == 'test-model'
    assert len(request['messages']) == 5
    assert 'Scene facts' in request['messages'][1]['content']


def test_unreachable_llm_reports_setup_and_preserves_history():
    from urllib.error import URLError
    chat = SceneChat({}, 'test')
    with patch('vision_memory.chat.urlopen', side_effect=URLError('offline')):
        with pytest.raises(RuntimeError, match='ollama serve'):
            chat.ask('What moved?')
    assert chat.history == []


def test_llm_context_preserves_all_measured_relations():
    facts = scene_facts(scene(('wallet', (100, 100, 200, 200)),
                              ('pen', (220, 100, 250, 200))))
    chat = SceneChat(facts, 'test')
    context = json.loads(chat.context[1]['content'].split('\n', 1)[1])
    columns = context['spatial_relations']['columns']
    rows = [dict(zip(columns, row)) for row in context['spatial_relations']['rows']]
    assert rows == [{'subject': 'wallet #1', 'object': 'pen #1',
                     'relations': ['left of', 'next to']}]
    assert context['comparison']['available'] is False


def test_direct_neighbor_list_is_complete_without_model_omissions():
    facts = {'next_to': {'cup #1': ['pen #1', 'book #1', 'book #2']}}
    chat = SceneChat(facts, 'test')
    with patch('vision_memory.chat.urlopen') as post:
        answer = chat.ask('What is next to cup #1?')
    post.assert_not_called()
    assert all(item in answer for item in facts['next_to']['cup #1'])
    assert chat.neighbor_list_target('What is not next to cup #1?') is None


def test_ambiguous_neighbor_target_is_left_to_conversation():
    chat = SceneChat({'next_to': {'pen #1': [], 'pen #2': []}}, 'test')
    assert chat.neighbor_list_target('What is next to the pen?') is None
    assert chat.neighbor_list_target('What is next to pen #2?') == 'pen #2'
