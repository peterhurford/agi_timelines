import yaml
import os
from datetime import datetime

# Load authoritative benchmark data from METR-Horizon-v1.1
# Performance values: p50/p80 horizon length in hours
# YAML source values are in minutes, converted to hours by dividing by 60
# p50 = task length at 50% reliability, p80 = task length at 80% reliability
# Note that 'reliability' is often confused - see https://x.com/peterwildeford/status/1967963942589747557

_YAML_PATH = os.path.join(os.path.dirname(__file__), 'benchmark_results_1_1.yaml')

# Map YAML keys to the model_data keys used throughout the codebase
_YAML_KEY_MAP = {
    'gpt2': 'gpt2',
    'davinci_002': 'gpt3',
    'gpt_3_5_turbo_instruct': 'gpt3p5_turbo',
    'gpt_4': 'gpt4',
    'gpt_4_1106_inspect': 'gpt4_nov23',
    'claude_3_opus_inspect': 'claude_3_opus',
    'gpt_4_turbo_inspect': 'gpt4_turbo',
    'gpt_4o_inspect': 'gpt4o',
    'claude_3_5_sonnet_20240620_inspect': 'claude_3p5_sonnet_old',
    'o1_preview': 'o1_preview',
    'claude_3_5_sonnet_20241022_inspect': 'claude_3p5_sonnet_new',
    'o1_inspect': 'o1',
    'claude_3_7_sonnet_inspect': 'claude_3p7_sonnet',
    'o3_inspect': 'o3',
    'claude_4_opus_inspect': 'claude_4_opus',
    'gpt_5_2025_08_07_inspect': 'gpt5',
    'gemini_3_pro': 'gemini_3',
    'gpt_5_1_codex_max_inspect': 'gpt5.1-codex-max',
    'claude_opus_4_5_inspect': 'claude_4p5_opus',
    'gpt_5_2': 'gpt5.2',
}

# Human-readable display names for each model
_DISPLAY_NAMES = {
    'gpt2': 'GPT-2',
    'gpt3': 'GPT-3 (davinci-002)',
    'gpt3p5_turbo': 'GPT-3.5 Turbo Instruct',
    'gpt4': 'GPT-4',
    'gpt4_nov23': 'GPT-4-Nov23',
    'claude_3_opus': 'Claude 3 Opus',
    'gpt4_turbo': 'GPT-4 Turbo',
    'gpt4o': 'GPT-4o',
    'claude_3p5_sonnet_old': 'Claude 3.5 Sonnet (old)',
    'o1_preview': 'o1 preview',
    'claude_3p5_sonnet_new': 'Claude 3.5 Sonnet (new)',
    'o1': 'o1',
    'claude_3p7_sonnet': 'Claude 3.7 Sonnet',
    'o3': 'o3',
    'claude_4_opus': 'Claude 4 Opus',
    'gpt5': 'GPT-5',
    'gemini_3': 'Gemini 3 Pro',
    'gpt5.1-codex-max': 'GPT-5.1 Codex Max',
    'claude_4p5_opus': 'Claude 4.5 Opus',
    'gpt5.2': 'GPT 5.2',
    # Models not in YAML benchmark
    'claude_4_sonnet': 'Claude 4 Sonnet',
    'gemini_2p5_pro': 'Gemini 2.5 Pro',
    'grok_4': 'Grok 4',
    'claude_4p1_opus': 'Claude 4.1 Opus',
    'claude_4p5_sonnet': 'Claude 4.5 Sonnet',
    'gpt5.2codex': 'GPT 5.2 Codex',
}


def _load_yaml_data():
    """Load benchmark results from YAML and build model_data dict."""
    with open(_YAML_PATH, 'r') as f:
        raw = yaml.safe_load(f)

    data = {}
    for yaml_key, result in raw['results'].items():
        model_key = _YAML_KEY_MAP.get(yaml_key)
        if model_key is None:
            continue

        metrics = result['metrics']
        release_date = result['release_date']
        if isinstance(release_date, str):
            release_date = datetime.strptime(release_date, '%Y-%m-%d')
        else:
            release_date = datetime(release_date.year, release_date.month, release_date.day)

        # YAML p50/p80 horizon lengths are in minutes; convert to hours
        p50 = metrics.get('p50_horizon_length', {}).get('estimate')
        p80 = metrics.get('p80_horizon_length', {}).get('estimate')

        data[model_key] = {
            'name': _DISPLAY_NAMES.get(model_key, yaml_key),
            'launch_date': release_date,
            'performance_50p': p50 / 60 if p50 is not None else None,
            'performance_80p': p80 / 60 if p80 is not None else None,
        }

    return data


def _build_model_data():
    """Build the complete model_data dict: YAML data + models not in YAML."""
    data = _load_yaml_data()

    # Models not present in the YAML benchmark but tracked in the codebase.
    # Performance values are from prior estimates (not METR-Horizon-v1.1).
    _extra_models = {
        'claude_4_sonnet': {
            'name': 'Claude 4 Sonnet',
            'launch_date': datetime(2025, 5, 22),
            'performance_50p': 1 + 15 / 60,
            'performance_80p': 17 / 60,
        },
        'gemini_2p5_pro': {
            'name': 'Gemini 2.5 Pro',
            'launch_date': datetime(2025, 6, 5),
            'performance_50p': 40 / 60,
            'performance_80p': 9 / 60,
        },
        'grok_4': {
            'name': 'Grok 4',
            'launch_date': datetime(2025, 7, 9),
            'performance_50p': 1 + 49 / 60,
            'performance_80p': 15 / 60,
        },
        'claude_4p1_opus': {
            'name': 'Claude 4.1 Opus',
            'launch_date': datetime(2025, 8, 5),
            'performance_50p': 1 + 54 / 60,
            'performance_80p': 22 / 60,
        },
        'claude_4p5_sonnet': {
            'name': 'Claude 4.5 Sonnet',
            'launch_date': datetime(2025, 9, 29),
            'performance_50p': 2 + 2 / 60,
            'performance_80p': 21 / 60,
        },
        'gpt5.2codex': {
            'name': 'GPT 5.2 Codex',
            'launch_date': datetime(2025, 12, 18),
            'performance_50p': None,
            'performance_80p': None,
        },
    }

    for key, val in _extra_models.items():
        if key not in data:
            data[key] = val

    return data


model_data = _build_model_data()
