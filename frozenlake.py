import csv
import random
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).parent
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
GOAL = 15
MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]

FIXED_POLICY = {
    0: DOWN, 1: LEFT, 2: DOWN, 3: LEFT,
    4: DOWN, 6: DOWN,
    8: RIGHT, 9: DOWN, 10: DOWN,
    13: RIGHT, 14: RIGHT,
}


def run_episode(env, policy, seed):
    rng = random.Random(seed)
    state, _ = env.reset(seed=seed)
    total, frames = 0, [env.render()]
    for step in range(1, MAX_STEPS + 1):
        action = rng.randrange(4) if policy == "random" else FIXED_POLICY[state]
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        frames.append(env.render())
        if terminated or truncated:
            break
    if terminated:
        outcome = "goal" if state == GOAL else "hole"
    else:
        outcome = "time limit"
    return step, total, outcome, frames


def main():
    rows, summary, traces = [], [], {}
    for slippery in [False, True]:
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery, render_mode="ansi")
        for policy in ["random", "fixed"]:
            condition = f"slippery={slippery}, {policy}"
            for batch in BATCH_SEEDS:
                results = []
                for i in range(50):
                    seed = 1000 * batch + i
                    steps, total, outcome, frames = run_episode(env, policy, seed)
                    results.append((steps, outcome))
                    rows.append([condition, batch, seed, policy, steps, total, outcome])
                    if slippery and policy == "fixed":
                        traces.setdefault(outcome, (seed, frames))
                goals = sum(o == "goal" for _, o in results)
                summary.append(f"| {slippery} | {policy} | {batch} | {goals} | "
                               f"{sum(o == 'hole' for _, o in results)} | "
                               f"{sum(o == 'time limit' for _, o in results)} | "
                               f"{goals * 2}% | {sum(s for s, _ in results) / 50:.2f} |")
        env.close()

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    with open(out / "part_b_episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "batch_seed", "episode_seed", "policy", "steps", "total_reward", "outcome"])
        writer.writerows(rows)
    table = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"] + summary
    (out / "part_b_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    with open(out / "part_b_traces.txt", "w", encoding="utf-8") as f:
        for outcome, (seed, frames) in traces.items():
            f.write(f"===== slippery fixed policy, episode seed {seed}, outcome: {outcome} =====\n")
            f.write("\n".join(frames) + "\n\n")
    print("\n".join(table))
    print("traces saved for:", list(traces))


if __name__ == "__main__":
    main()
