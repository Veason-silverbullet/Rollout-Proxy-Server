"""Raw recording integrity, optional training fields, and conversation resets."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxyserver.record_validation import validate_record_structure


def record():
    return {'turns': [{'agent_id': 'agent', 'prompt_token_ids': [1, 2],
        'completion_token_ids': [3, 4], 'completion_logprobs': [-.1, -.2],
        'new_conversation': True, 'sampling_params': {'top_p': .9},
        'rollout_top_p_token_ids': [3, 7, 4], 'rollout_top_p_token_offsets': [0, 2, 3]}],
        'routed_experts': {'agent': {'rows': 3, 'cols': 1, 'data': 'AQID'}}}


def rejects(data, **kwargs):
    try:
        validate_record_structure(data, **kwargs)
    except ValueError:
        return
    raise AssertionError(f'Accepted malformed record: {data}')


def test_integrity():
    data = record()
    original = deepcopy(data)
    validate_record_structure(data, require_logprobs=True, require_sampling_masks=True, require_routed_experts=True)
    assert data == original
    for field, values in {
        'prompt_token_ids': [[True], [-1], [1.0], None],
        'completion_token_ids': [[False], [-1], ['3'], None],
        'completion_logprobs': [[], [0.1], [float('nan'), 0], [float('inf'), 0], [True, 0], None],
        'new_conversation': [1, 'true'],
        'sampling_params': [{'top_p': True}, {'top_p': float('inf')}, {'top_p': 10**999}],
        'rollout_top_p_token_ids': [None, [True, 7, 4], [8, 7, 4], [3, 3, 4]],
        'rollout_top_p_token_offsets': [None, [], [0, 0, 3], [0, 4, 3], [0, 2, 4], [False, 2, 3]],
    }.items():
        for value in values:
            changed = record()
            changed['turns'][0][field] = value
            rejects(changed)
    for field, values in {'rows': [True, -1, 4], 'cols': [0, True], 'start': [1, False],
                          'dtype': ['int32'], 'data': ['!', 'AQI=', None, 4]}.items():
        for value in values:
            changed = record()
            changed['routed_experts']['agent'][field] = value
            rejects(changed)


def test_optional_fields_and_segments():
    data = record()
    for field in ('completion_logprobs', 'sampling_params', 'rollout_top_p_token_ids', 'rollout_top_p_token_offsets'):
        data['turns'][0].pop(field)
    data.pop('routed_experts')
    # Persisted files may omit optional fields. Direct engines need no version.
    validate_record_structure(data)
    for requirement in ('require_logprobs', 'require_sampling_masks', 'require_routed_experts'):
        rejects(data, **{requirement: True})
        validate_record_structure(data, selected={'other'}, **{requirement: True})
        validate_record_structure({'turns': []}, **{requirement: True})
    data = record()
    data['turns'].append(deepcopy(data['turns'][0]))
    validate_record_structure(data, require_logprobs=True)  # two separate conversations
    rejects(data, require_routed_experts=True)  # only last conversation captured
    data['turns'][1]['new_conversation'] = False
    validate_record_structure(data)  # partial capture is legitimate
    rejects(data, require_routed_experts=True)
    data['routed_experts']['agent'].update(rows=7, data='AQIDBAUGBw==')
    validate_record_structure(data, require_routed_experts=True)


if __name__ == '__main__':
    test_integrity()
    test_optional_fields_and_segments()
    print('PASS raw record integrity and optional training requirements')
