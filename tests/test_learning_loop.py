"""Exercise the real coordinator with a fake service and fake MineRL boundary."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
from optimus1.learn import run, load_tasks

class LearningLoopTests(unittest.TestCase):
    def test_failure_feedback_memory_reuse_resume(self):
        task = {'id': 'wood', 'instruction': 'chop a tree', 'target': 'logs', 'quantity': 1}
        second = {'id': 'planks', 'instruction': 'craft planks', 'target': 'planks', 'quantity': 1}
        plan = [{'task': 'chop a tree', 'goal': ['logs', 1]}]
        calls, envs = [], []
        class Logger:
            def info(self, *a): pass
            def exception(self, *a): pass
        class Env:
            api_thread = None
            closed = False
            def __init__(self):
                self.env = self
                envs.append(self)
            def reset(self): return {'inventory': {}, 'pov': None}
            def noop_action(self): return {}
            def step(self, action):
                inv = {} if len(envs) == 1 else {'oak_log': 1, 'oak_planks': 1}
                return {'inventory': inv}, 0, False, {}
            def close(self): self.closed = True
            def save_video(self, *a): return None
        class Memory:
            def __init__(self, cfg, logger):
                self.reflection_dir_path = cfg.memory.path + '/reflection'
                self.saved = []
            def retrieve_plan(self, instruction):
                return (json.dumps(self.saved) if self.saved else None), False
            def retrieve_graph(self, *a): return 'graph'
            def _save_plan(self, *a):
                if a[3] == 'success': self.saved.append({'task': a[0], 'plan': a[4]})
        class API:
            @staticmethod
            def _reset(cfg): pass
            @staticmethod
            def get_retrieval(*a): return 'visual'
            @staticmethod
            def get_plan(cfg, obs, instruction, **kwargs):
                calls.append((instruction, kwargs))
                if instruction == 'Select next learning task':
                    chosen = 'wood' if len(envs) < 3 else 'planks'
                    return json.dumps({'step 1': {'task': chosen}})
                current = plan if instruction == 'chop a tree' else [{'task': 'craft planks', 'goal': ['planks', 1]}]
                return json.dumps({'step 1': current[0]})
        class Monitor:
            def __init__(self, *a): pass
            def get_metric(self): return {}
        modules = {}
        def module(name, **attrs):
            m = types.ModuleType(name)
            m.__dict__.update(attrs)
            modules[name] = m
        module('optimus1.main', EXAMPLE='example', get_info_from_plan=lambda x: ('logs', 'forest', 'forest'),
               agent_do=lambda cfg, env, log, mon, p, obs, mem: ('success', 5, p))
        module('optimus1.env', env_make=lambda *a: Env(), register_custom_env=lambda cfg: None)
        module('optimus1.memories', Memory=Memory)
        module('optimus1.monitor', Monitors=Monitor, SuccessMonitor=Monitor, StepMonitor=Monitor)
        module('optimus1.util', ServerAPI=API, get_logger=lambda x: Logger(),
               render_gpt4_plan=lambda x: list(json.loads(x).values()))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            cfg = OmegaConf.create({'learning': {'directory': tmp, 'resume': False,
                'max_iterations': 3, 'max_attempts_per_task': 3, 'selection_retries': 2,
                'tasks': [task, second]}, 'memory': {'path': tmp+'/memory'},
                'env': {'name': 'fake'}, 'server': {}})
            run(cfg)
            state = json.loads((Path(tmp)/'state.json').read_text())
            self.assertEqual([a['success'] for a in state['attempts']], [False, True, True])
            self.assertNotIn('skills', state)
            self.assertEqual(sum(a['success'] for a in state['attempts']), 2)
            self.assertTrue(all(e.closed for e in envs))
            planning_calls = [k for i, k in calls if i != 'Select next learning task']
            self.assertIn('final_inventory', planning_calls[1]['example'])
            self.assertIn('chop a tree', planning_calls[2]['example'])
            cfg.learning.resume = True
            run(cfg)
            self.assertEqual(len(envs), 3)

    def test_pool_without_game_imports(self):
        cfg = OmegaConf.create({'learning': {'tasks': [], 'groups': ['wooden', 'iron']}})
        tasks = load_tasks(cfg)
        self.assertTrue(any(t['target'] == 'iron_ore' for t in tasks))
        self.assertEqual(len({t['id'] for t in tasks}), len(tasks))

if __name__ == '__main__': unittest.main()
