"""Voyager-style bounded automatic curriculum using the existing model service."""
import copy
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from optimus1.evolution import LearningState, validate_plan


def load_tasks(cfg):
    if cfg.learning.tasks:
        tasks = OmegaConf.to_container(cfg.learning.tasks, resolve=True)
    else:
        tasks = []
        for group in cfg.learning.groups:
            path = Path(__file__).parent / 'conf' / 'benchmark' / f'{group}.yaml'
            for entry in OmegaConf.load(path).all_task:
                instruction = entry.instruction.strip()
                from optimus1.evolution import task_target
                target = task_target(instruction)
                tasks.append({'id': f'{group}:{entry.id}', 'instruction': instruction,
                              'target': target, 'quantity': 1})
    for task in tasks:
        if not all(k in task for k in ('id', 'instruction', 'target', 'quantity')):
            raise ValueError('Tasks require id, instruction, target, quantity')
        if not isinstance(task['quantity'], int) or task['quantity'] < 1:
            raise ValueError('Task quantity must be a positive integer')
    return tasks


def run(cfg):
    # Imported lazily so --cfg job works without Minecraft/model dependencies.
    import optimus1.main as runtime
    from optimus1.env import env_make, register_custom_env
    from optimus1.memories import Memory
    from optimus1.monitor import Monitors, StepMonitor, SuccessMonitor
    from optimus1.util import ServerAPI, get_logger, render_gpt4_plan

    if cfg.learning.max_attempts_per_task < 1 or cfg.learning.max_iterations < 1 or cfg.learning.selection_retries < 1:
        raise ValueError('Learning limits must be positive')
    tasks = load_tasks(cfg)
    root = Path(cfg.learning.directory)
    state = LearningState(root / 'state.json', tasks, cfg.learning.resume)
    logger = get_logger(__name__)
    memory = Memory(cfg, logger)
    runtime.REFLECTION_IMAGE_ROOT = str(Path(memory.reflection_dir_path) / 'img')
    register_custom_env(cfg)

    while len(state.data['attempts']) < cfg.learning.max_iterations:
        candidates = state.candidates(cfg.learning.max_attempts_per_task)
        if not candidates:
            logger.info('No eligible tasks remain; learning stopped.')
            break
        env = env_make(cfg.env.name, cfg, logger)
        task = None
        plan, inventory, steps = [], {}, 0
        feedback, success = '', False
        try:
            # Synchronous reset propagates service errors instead of losing them in a thread.
            ServerAPI._reset(cfg.server)
            obs = env.reset()
            history = [{k: a[k] for k in ('task_id', 'success', 'feedback')}
                       for a in state.data['attempts'][-20:]]
            prompt = (
                'Select exactly ONE achievable, useful next Minecraft learning task. '
                'Prefer unexplored tasks and prerequisites for harder tasks; avoid repeatedly failing tasks. '
                'Use the screenshot, inventory and learning history. Return <task planning> followed '
                'by JSON with exactly "step 1": {"task": the selected candidate ID, '
                '"goal": [its target, its quantity]}. Do not execute or decompose it.\n'
                + json.dumps({'candidates': candidates, 'history': history,
                              'inventory': {k: int(v) for k, v in obs['inventory'].items() if v}},
                             ensure_ascii=False)
            )
            selection_error = None
            for _ in range(cfg.learning.selection_retries):
                try:
                    response = ServerAPI.get_plan(cfg.server, obs, 'Select next learning task',
                        example=prompt, visual_info=prompt, graph='Use only the listed candidate IDs.')
                    task = state.choose(response, candidates)
                    break
                except ValueError as exc:
                    selection_error = exc
            if task is None:
                raise RuntimeError(f'Invalid curriculum responses: {selection_error}')
            logger.info('Learning task: %s', task['instruction'])
            info = ServerAPI.get_retrieval(cfg.server, obs, task['instruction'])
            _, visual, environment = runtime.get_info_from_plan(info)
            memory.current_environment = environment
            example, _ = memory.retrieve_plan(task['instruction'])
            previous = [a for a in state.data['attempts'] if a['task_id'] == task['id']][-1:]
            context = json.dumps({'previous_failed_attempt': previous}, ensure_ascii=False)
            raw = ServerAPI.get_plan(cfg.server, obs, task['instruction'],
                example=(example or runtime.EXAMPLE) + '\nAdapt the reference plan and correct previous failures:\n' + context,
                graph=memory.retrieve_graph(task['target'], task['quantity']),
                visual_info=visual + '\nA fresh episode has started. Final goal must be '
                            + json.dumps([task['target'], task['quantity']]))
            plan = render_gpt4_plan(raw)
            validate_plan(plan, task['target'], task['quantity'])
            monitors = Monitors([SuccessMonitor(), StepMonitor()])
            status, steps, executed = runtime.agent_do(cfg, env, logger, monitors,
                                                       copy.deepcopy(plan), obs, memory)
            # Obtain the actual end state; never trust plan exhaustion as task success.
            final_obs, _, _, _ = env.step(env.env.noop_action())
            inventory = {k: int(v) for k, v in final_obs['inventory'].items() if v}
            feedback = json.dumps({'executor_status': status, 'metrics': monitors.get_metric(),
                                  'final_inventory': inventory, 'executed_plan': executed}, default=str)
            success = status == 'success'
            if success:
                validate_plan(executed, task['target'], task['quantity'])
                plan = executed
            from optimus1.evolution import inventory_count
            success = success and inventory_count(inventory, task['target']) >= task['quantity']
            video = env.save_video(task['instruction'], 'success' if success else 'failed')
            # Synchronous writer ensures memory is visible before the next task.
            memory._save_plan(task['instruction'], visual, task['target'],
                              'success' if success else 'failed', executed, steps, video, environment)
        except Exception as exc:
            if task is None:
                raise  # Infrastructure/selection failure must not silently become a learning task.
            feedback = f'{type(exc).__name__}: {exc}'
            success = False
            logger.exception('Learning attempt failed')
        finally:
            try:
                if env.api_thread is not None:
                    env.api_thread.join()
            finally:
                env.close()
        state.record(task, success, plan, feedback, inventory, steps)
        logger.info('Saved attempt %s; completed tasks: %s',
                    len(state.data['attempts']), len(state.completed_tasks))


@hydra.main(version_base=None, config_path='conf', config_name='learn')
def main(cfg):
    run(cfg)


if __name__ == '__main__':
    main()
