# Automatic curriculum learning for Optimus-1

The learner borrows Voyager's task selection and feedback loop, while keeping Optimus's existing plan, reflection and replan memory. There is no separate skill library or new action representation. It uses the running `/chat` and `/reset` service unchanged.

## Run and resume

Start the existing model service, then run from the repository root in your Optimus environment:

```bash
bash scripts/learn.sh server.port=9000 learning.groups='[wooden]' learning.max_iterations=3 learning.directory=outputs/learning/smoke
```

Resume with the SAME task pool and directory:

```bash
bash scripts/learn.sh server.port=9000 learning.groups='[wooden]' learning.max_iterations=20 learning.directory=outputs/learning/smoke learning.resume=true
```

`max_iterations` includes previously finished attempts. An existing checkpoint requires `resume=true`; a new run requires a new directory. Only one learner may use an output directory and model service at a time.

## What changes between tasks

1. Reset the controller service and start a fresh MineRL episode.
2. The existing planning model selects an eligible task using current observations and recent learning results. Invalid selections are retried up to `selection_retries` times.
3. Retrieve a successful reference plan through the existing `Memory.retrieve_plan()`. Include the last failure for this task when generating a new plan.
4. Validate the plan's final objective and run the existing `agent_do()` executor, including its internal replanning.
5. Verify that the final inventory contains the requested item quantity AND the executor reports success.
6. Store the execution in the original Optimus plan memory. Record attempt feedback and progress in `state.json`, then select another task.

Failed tasks remain eligible until `max_attempts_per_task` is reached; there is no forced immediate retry. Successful tasks leave the pool. The existing reflection and replan memories remain available. Plan retrieval uses the configured directory, and reflection retrieval refreshes its list of files.

## Configuration and output

See `src/optimus1/conf/learn.yaml`. Defaults: all seven benchmark configuration groups, 160 total attempts, four attempts per task, three selection retries, empty initial inventory, and 20 minutes per episode. `evaluate` does not filter the learning pool.

An explicit task pool can replace the benchmark groups:

```bash
bash scripts/learn.sh learning.directory=outputs/learning/custom 'learning.tasks=[{id:logs,instruction:"chop a tree",target:logs,quantity:2}]'
```

Under `learning.directory`:

- `state.json`: task definitions, attempt history, feedback and final inventories. Completed tasks are derived from successful attempts. This is progress bookkeeping, not a second memory/skill library.
- `memory/`: the original Optimus successful/failed plan, reflection and replan records.
- `videos/` and `logs/`: existing recording and Hydra outputs.

Checkpoint writes are atomic. An interrupted attempt may be repeated after resume; the Minecraft world is not restored. Legacy schema-1 progress files are accepted and their redundant `skills` field is removed on resume; the original plan memory files remain the source for retrieval.

## Scope and validation

This is a bounded automatic curriculum, with each attempt starting a fresh environment. It retains the original preloaded recipe graph and crafting helpers. It does not discover recipes online, train model parameters, generate JavaScript skills, or introduce a separate skill store. The final checker supports inventory-acquisition objectives.

The default learning pool includes benchmark tasks. Evaluating those same tasks afterward is not held-out generalization; use an explicitly disjoint pool when evaluating transfer.

CPU tests exercise the coordinator with mocked model/MineRL boundaries, existing-memory reuse, failure feedback, inventory verification, attempt limits and resume. Real Minecraft/model-service execution has not been tested here.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m optimus1.learn --cfg job
```
