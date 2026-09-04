import json
import re
from typing import Any, Dict, List

PlanList = List[Dict[str, Any]]

_ITEM_ALIASES = {
    "log": "logs",
    "oak_log": "logs",
    "birch_log": "logs",
    "spruce_log": "logs",
    "jungle_log": "logs",
    "acacia_log": "logs",
    "dark_oak_log": "logs",
    "oak_logs": "logs",
    "tree": "logs",
    "stick": "stick",
    "sticks": "stick",
    "plank": "planks",
    "oak_planks": "planks",
    "birch_planks": "planks",
    "wooden_planks": "planks",
}


def normalize_item(name: str) -> str:
    item = str(name).strip().lower().replace(" ", "_")
    return _ITEM_ALIASES.get(item, item)


def task_target_item(task: str) -> str:
    text = " ".join(str(task).lower().split())
    prefixes = (
        "smelt and craft a ",
        "smelt and craft an ",
        "smelt and craft ",
        "craft a ",
        "craft an ",
        "craft ",
        "smelt a ",
        "smelt an ",
        "smelt ",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return normalize_item(text[len(prefix) :])
    if "chop" in text and "tree" in text:
        return "logs"
    if "dirt" in text:
        return "dirt"
    return normalize_item(text.split()[-1]) if text else ""


def plan_step_item(step: Dict[str, Any]) -> str:
    goal = step.get("goal")
    if isinstance(goal, (list, tuple)) and goal:
        return normalize_item(goal[0])
    if isinstance(goal, str):
        return normalize_item(goal)
    return ""


def plan_ends_with_task(planning: PlanList, task: str) -> bool:
    if not planning:
        return False
    target = task_target_item(task)
    return bool(target) and plan_step_item(planning[-1]) == target


def trim_plan_to_task(planning: PlanList, task: str) -> PlanList:
    """Keep steps through the first goal that matches the evaluate task."""
    target = task_target_item(task)
    if not planning or not target:
        return planning
    for idx, step in enumerate(planning):
        if plan_step_item(step) == target:
            return planning[: idx + 1]
    return planning


def render_gpt4_plan(plan: str, is_replan: bool = False) -> PlanList:
    plan = plan.replace("<task planning>:", "<task planning>").replace("**", "")
    sep_str = "<replan>:" if is_replan else "<task planning>"

    temp = plan.split(sep_str)[-1].strip()
    if "```json" in temp:
        temp = temp.split("```json")[1].strip().split("```")[0].strip()

    if "{{" in temp:
        temp = temp.replace("{{", "{").replace("}}", "}")

    r = temp.rfind("}")
    temp = temp[: r + 1]

    temp = json.loads(temp)

    sub_plans = [
        temp[step]
        for step in temp.keys()
        if "open" not in temp[step]["task"]
        and "place" not in temp[step]["task"]
        and "access" not in temp[step]["task"]
    ]

    for p in sub_plans:
        p["task"] = p["task"].replace("punch", "chop").replace("collect", "chop").replace("gather", "chop")

    return sub_plans


def render_reflection(reflection: str):
    """Environment: <Ocean>
    Situation: <Replan>
    Predicament: <In_water>
    """
    reflection = reflection.strip().replace(": ", ": <")

    matches = re.findall(r"<([^<]+)$", reflection, re.MULTILINE)
    rp = None
    if len(matches) == 3:
        rp = matches[2].split("/")[0].strip().lower()
    res = (
        matches[0].split("/")[0].replace(">", "").strip().lower().split(" ")[0].split("\n")[0],
        matches[1].split("/")[0].replace(">", "").strip().lower().split(" ")[0].split("\n")[0],
        rp,
    )
    return res


def render_recipe(recipe) -> str:
    lst = [f'"{k}": {v}' for k, v in recipe.items()]

    return "{" + ", ".join(lst) + "}"


def render_replan_example(replan: List[Dict[str, Any]]):
    res = {}
    for idx, plan in enumerate(replan):
        res[f"step {idx + 1}"] = plan
    return json.dumps(res)


if __name__ == "__main__":
    plan = """{
    "step 1": {"task": "punch a tree", "goal": ["logs", 3]},
    "step 2": {"task": "open inventory", "goal": ["inventory accessed", 1]},
    "step 3": {"task": "craft planks", "goal": ["planks", 12]},
    "step 4": {"task": "craft sticks", "goal": ["sticks", 4]},
    "step 5": {"task": "place crafting table", "goal": ["crafting_table placed", 1]},
    "step 6": {"task": "use crafting table", "goal": ["crafting_table used", 1]},
    "step 7": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]}
}"""
    temp = json.loads(plan)
    print(temp["step 1"]["task"])
    sub_plans = [
        temp[step] for step in temp.keys() if "open" not in temp[step]["task"] and "place" not in temp[step]["task"]
    ]
    print(sub_plans)
    # print(render_reflection(plan))
