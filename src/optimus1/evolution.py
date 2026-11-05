"""Dependency-free curriculum bookkeeping and learning progress."""
import json
import math
import re
from pathlib import Path


def parse_object(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    raise ValueError('Model did not return a JSON object')


def validate_plan(plan, target, quantity=1):
    if not isinstance(plan, list) or not plan:
        raise ValueError('Empty plan')
    for step in plan:
        if not isinstance(step, dict) or not isinstance(step.get('task'), str):
            raise ValueError('Invalid plan step')
        goal = step.get('goal')
        if not isinstance(goal, (list, tuple)) or len(goal) != 2:
            raise ValueError('Each goal must be [item, quantity]')
        if not isinstance(goal[0], str) or not isinstance(goal[1], (int, float)):
            raise ValueError('Invalid goal')
        if isinstance(goal[1], bool) or not math.isfinite(goal[1]) or goal[1] <= 0:
            raise ValueError('Goal quantity must be positive')
    if plan[-1]['goal'][0] != target or plan[-1]['goal'][1] < quantity:
        raise ValueError('Plan does not end at the requested objective')


def inventory_count(inventory, target):
    aliases = {'logs': ('_log',), 'planks': ('_planks',)}
    if target in aliases:
        return sum(int(v) for k, v in inventory.items()
                   if k == target or any(k.endswith(s) for s in aliases[target]))
    return int(inventory.get(target, 0))


class LearningState:
    def __init__(self, path, tasks, resume=False):
        self.path = Path(path)
        self.tasks = {t['id']: t for t in tasks}
        if len(self.tasks) != len(tasks) or not tasks:
            raise ValueError('Task pool must be nonempty with unique IDs')
        if self.path.exists():
            if not resume:
                raise FileExistsError('Checkpoint exists; use learning.resume=true or a new directory')
            self.data = json.loads(self.path.read_text())
            if self.data.get('schema') not in (1, 2) or self.data['tasks'] != tasks:
                raise ValueError('Checkpoint schema/task pool mismatch')
            # Legacy checkpoints already contain the full attempt history.
            self.data.pop('skills', None)
            self.data['schema'] = 2
            self.save()
        else:
            if resume:
                raise FileNotFoundError('Cannot resume a missing checkpoint')
            self.data = {'schema': 2, 'tasks': tasks, 'attempts': []}
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def candidates(self, max_attempts):
        counts = {}
        for a in self.data['attempts']:
            counts[a['task_id']] = counts.get(a['task_id'], 0) + 1
        return [t for t in self.tasks.values() if t['id'] not in self.completed_tasks
                and counts.get(t['id'], 0) < max_attempts]

    def choose(self, response, candidates):
        obj = parse_object(response)
        # Use the existing planner's step/goal response schema.
        step = obj.get('step 1', {})
        chosen = step.get('task')
        for task in candidates:
            if chosen in (task['id'], task['instruction']):
                return task
        raise ValueError('Curriculum selected a task outside the eligible pool')

    @property
    def completed_tasks(self):
        return {a['task_id'] for a in self.data['attempts'] if a['success']}

    def record(self, task, success, plan, feedback, inventory, steps=0):
        # Independent inventory verification prevents executor false positives.
        success = bool(success and inventory_count(inventory, task['target']) >= task['quantity'])
        if success:
            validate_plan(plan, task['target'], task['quantity'])
        self.data['attempts'].append({'task_id': task['id'], 'success': success,
            'feedback': feedback, 'inventory': inventory, 'plan': plan, 'steps': steps})
        self.save()
        return success


def task_target(instruction):
    text = instruction.lower().strip()
    for prefix in ('smelt and craft ', 'craft ', 'smelt '):
        if text.startswith(prefix):
            text = re.sub(r'^(?:a|an) ', '', text[len(prefix):])
            return text.replace(' ', '_')
    if 'tree' in text: return 'logs'
    for phrase, item in [('iron ore', 'iron_ore'), ('gold ore', 'gold_ore')]:
        if phrase in text: return item
    return text.split()[-1]
