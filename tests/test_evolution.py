import json
import tempfile
import unittest
from pathlib import Path
from optimus1.evolution import LearningState, inventory_count, task_target, validate_plan

TASK = {'id': 'wood', 'instruction': 'chop a tree', 'target': 'logs', 'quantity': 1}
PLAN = [{'task': 'chop a tree', 'goal': ['logs', 1]}]

class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'state.json'
        self.state = LearningState(self.path, [TASK])

    def test_failure_then_success_and_resume(self):
        self.assertFalse(self.state.record(TASK, True, PLAN, 'empty inventory', {}, 1))
        self.assertEqual(self.state.completed_tasks, set())
        self.assertTrue(self.state.record(TASK, True, PLAN, 'done', {'oak_log': 1}, 2))
        restored = LearningState(self.path, [TASK], resume=True)
        self.assertEqual(len(restored.data['attempts']), 2)
        self.assertEqual(restored.candidates(4), [])
        self.assertEqual(restored.completed_tasks, {'wood'})
        self.assertNotIn('skills', restored.data)

    def test_attempt_limit_and_overwrite_protection(self):
        self.state.record(TASK, False, [], 'timeout', {})
        self.assertEqual(self.state.candidates(1), [])
        with self.assertRaises(FileExistsError): LearningState(self.path, [TASK])
        with self.assertRaises(ValueError):
            LearningState(self.path, [dict(TASK, quantity=2)], resume=True)

    def test_curriculum_validation(self):
        self.assertEqual(self.state.choose('```json\n{"step 1":{"task":"wood"}}\n```', [TASK]), TASK)
        with self.assertRaises(ValueError):
            self.state.choose('{"step 1":{"task":"unknown"}}', [TASK])

    def test_invalid_plans_never_marked_complete(self):
        for plan in ([], [{'task': 'craft dirt', 'goal': ['dirt', 1]}],
                     [{'task': 'chop tree', 'goal': ['logs', -1]}]):
            with self.assertRaises(ValueError):
                self.state.record(TASK, True, plan, '', {'oak_log': 1})
        self.assertEqual(self.state.completed_tasks, set())

    def test_inventory_aliases_and_targets(self):
        self.assertEqual(inventory_count({'oak_log': 2, 'birch_log': 1, 'stick': 5}, 'logs'), 3)
        self.assertEqual(task_target('dig down to mine iron ore'), 'iron_ore')
        self.assertEqual(task_target('Craft a iron pickaxe'), 'iron_pickaxe')
        self.assertEqual(task_target('Smelt and craft a gold ingot'), 'gold_ingot')

if __name__ == '__main__': unittest.main()
