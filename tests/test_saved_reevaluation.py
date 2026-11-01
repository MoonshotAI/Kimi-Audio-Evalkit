"""Re-evaluate saved inference results without loading an audio model."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from almeval.datasets import build_dataset
from almeval.utils import dump
from almeval.utils.config_manager import ConfigManager


class TestSavedReevaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'config.yaml'
        self.env = patch.dict(os.environ, {'PROJECT_CONFIG_PATH': str(self.config)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.singleton = patch.object(ConfigManager, '_instance', None)
        self.singleton.start()
        self.addCleanup(self.singleton.stop)

    def saved_dataset(self, name, rows):
        path = self.root / f'fixture_{name}.jsonl'
        # VoiceBenchDownloader keeps the spoken instruction in audio_content;
        # merge_one_dataset adds real_prompt, which may be empty for audio input.
        records = [dict({'index': i, 'question': '', 'audio_content': f'Question {i}',
                         'audio_path': 'unused.wav', 'real_prompt': ''}, **row)
                   for i, row in enumerate(rows)]
        dump(records, str(path))
        self.config.write_text(json.dumps({'DATASETS': {
            'dataset_root': str(self.root), 'datasets': {name: str(path)}}}), encoding='utf-8')
        return build_dataset(name), str(path)

    def test_ifeval_uses_distinct_spoken_prompts_and_returns_metrics(self):
        dataset, path = self.saved_dataset('ifeval', [
            dict(subset='ifeval', audio_content='Include orange in your response.',
                 prediction='orange', instruction=['keywords:existence'],
                 instruction_kwargs=[{'keywords': ['orange']}]),
            dict(subset='ifeval', audio_content='Include banana in your response.',
                 prediction='pear', instruction=['keywords:existence'],
                 instruction_kwargs=[{'keywords': ['banana']}]),
        ])
        for dump_judge in (False, True):
            with self.subTest(dump_judge=dump_judge):
                result = dataset.evaluate(path, method='vb-ifeval', dump_judge=dump_judge)
                metrics = result['performance']['ifeval']
                for key in ('strict-prompt', 'strict-instruction', 'loose-prompt',
                            'loose-instruction', 'final'):
                    self.assertEqual(metrics[key], 0.5)
                self.assertFalse(Path(path.replace('.jsonl', '_vb-ifeval_judge.jsonl')).exists())

    def test_refqa_passes_the_spoken_question_to_the_metric(self):
        for name, question in (('sd-qa', 'Question 0'), ('ClothoAQA', 'Written question')):
            with self.subTest(dataset=name), patch.object(ConfigManager, '_instance', None):
                dataset, path = self.saved_dataset(name, [
                    dict(subset=name, question='Written question',
                         answer='orange', prediction='orange'),
                ])
                dataset.pedant = Mock()
                dataset.pedant.evaluate.return_value = True
                metrics, _ = dataset.evaluate_vb_qa(path)
                dataset.pedant.evaluate.assert_called_once_with(['orange'], 'orange', question)
                self.assertEqual(metrics[name]['acc'], 100)

    def test_advbench_judge_artifact_keeps_every_subset(self):
        dataset, path = self.saved_dataset('advbench', [
            dict(subset='first', prediction='I cannot help with that.'),
            dict(subset='second', prediction='Here is an answer.'),
        ])
        result = dataset.evaluate(path, method='vb-advbench')
        self.assertEqual(result['performance']['first']['correct'], 1)
        self.assertEqual(result['performance']['second']['correct'], 0)
        saved = pd.read_json(path.replace('.jsonl', '_vb-advbench_judge.jsonl'), lines=True)
        self.assertEqual(set(saved['subset']), {'first', 'second'})
        self.assertEqual(set(saved['index']), {0, 1})


if __name__ == '__main__':
    unittest.main()
