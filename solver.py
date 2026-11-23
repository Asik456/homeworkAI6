import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "data" / "maps.json").read_text(encoding="utf-8"))
TILES = CONFIG["tiles"]
GAMMA = CONFIG["discount"]
ACTIONS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
SIDEWAYS = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"]}


def all_states(grid):
    return [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != "#"]


def is_terminal(grid, s):
    return TILES[grid[s[0]][s[1]]].get("terminal", False)


def move(grid, s, direction):
    r, c = s[0] + ACTIONS[direction][0], s[1] + ACTIONS[direction][1]
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "#":
        return (r, c)
    return s


def outcomes(grid, s, action):
    rule = TILES[grid[s[0]][s[1]]]
    if rule["movement"] == "slippery":
        options = [(rule["intended"], action)] + [(rule["perpendicular_each"], d) for d in SIDEWAYS[action]]
    else:
        options = [(1.0, action)]
    result = []
    for p, direction in options:
        nxt = move(grid, s, direction)
        tile = TILES[grid[nxt[0]][nxt[1]]]
        result.append((p, nxt, tile["reward"], tile.get("terminal", False)))
    return result


def q_value(grid, V, s, action):
    return sum(p * (r + (0 if done else GAMMA * V[n])) for p, n, r, done in outcomes(grid, s, action))


def one_round(grid, V):
    new = dict(V)
    for s in all_states(grid):
        if not is_terminal(grid, s):
            new[s] = max(q_value(grid, V, s, a) for a in ACTIONS)
    return new


def value_iteration(grid, tolerance, max_iterations):
    V = {s: 0.0 for s in all_states(grid)}
    for i in range(1, max_iterations + 1):
        new = one_round(grid, V)
        change = max(abs(new[s] - V[s]) for s in V)
        V = new
        if change < tolerance:
            return V, i, change, True
    return V, max_iterations, change, False


def best_actions(grid, V, s):
    q = {a: q_value(grid, V, s, a) for a in ACTIONS}
    top = max(q.values())
    return [a for a in ACTIONS if abs(q[a] - top) < 1e-9]


def make_policy(grid, V):
    return {s: best_actions(grid, V, s)[0] for s in all_states(grid) if not is_terminal(grid, s)}


def value_map(grid, V):
    lines = []
    for r in range(len(grid)):
        cells = []
        for c in range(len(grid[0])):
            tile = grid[r][c]
            cells.append(f"{'#':>8}" if tile == "#" else f"{tile + ' ' + format(V[(r, c)], '.2f'):>8}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def policy_map(grid, V):
    lines = []
    for r in range(len(grid)):
        cells = []
        for c in range(len(grid[0])):
            s = (r, c)
            if grid[r][c] == "#" or is_terminal(grid, s):
                cells.append(f"{grid[r][c]:>6}")
            else:
                cells.append(f"{''.join(ARROWS[a] for a in best_actions(grid, V, s)):>6}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def find_start(grid):
    return next((r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == "S")


def run_episode(grid, policy, seed, max_steps):
    rng = random.Random(seed)
    s = find_start(grid)
    discounted = 0.0
    for step in range(max_steps):
        roll, total = rng.random(), 0.0
        for p, nxt, reward, done in outcomes(grid, s, policy[s]):
            total += p
            if roll < total:
                break
        discounted += (GAMMA ** step) * reward
        s = nxt
        if done:
            return step + 1, discounted, "goal" if grid[s[0]][s[1]] == "G" else "hole"
    return max_steps, discounted, "time limit"


def checks(grid):
    lines = ["Rule checks:"]
    lines.append(f"  wall: from (3,2) up -> {outcomes(grid, (2, 1), 'up')[0][1]} (stays, (2,2) is a wall)")
    lines.append(f"  boundary: from (1,1) left -> {outcomes(grid, (0, 0), 'left')[0][1]} (stays)")
    for a in ACTIONS:
        lines.append(f"  ice (2,3) {a}: probabilities sum = {sum(p for p, *_ in outcomes(grid, (1, 2), a)):.1f}")
    lines.append(f"  mud: (1,1) right -> reward {outcomes(grid, (0, 0), 'right')[0][2]}")
    lines.append(f"  goal: (1,3) right -> reward {outcomes(grid, (0, 2), 'right')[0][2]}, terminal = {outcomes(grid, (0, 2), 'right')[0][3]}")
    return "\n".join(lines)


def main():
    out = []
    sim = CONFIG["simulation"]
    rows = []
    for name in ["paper_map", "extended_map"]:
        grid = [list(row) for row in CONFIG[name]]
        out.append(f"===== {name} =====")
        if name == "paper_map":
            V0 = {s: 0.0 for s in all_states(grid)}
            V1 = one_round(grid, V0)
            out.append("V1:\n" + value_map(grid, V1))
            for r, c in CONFIG["paper_v2_positions"]:
                s = (r - 1, c - 1)
                q = {a: round(q_value(grid, V1, s, a), 4) for a in ACTIONS}
                out.append(f"V2 at ({r},{c}): {q} -> max {max(q.values())}")
            out.append(checks(grid))
        else:
            out.append("Sand (3,1) right -> reward " + str(outcomes(grid, (2, 0), "right")[0][2]))
            out.append("Water: (3,3) right -> reward " + str(outcomes(grid, (2, 2), "right")[0][2]))

        V, iterations, change, converged = value_iteration(grid, CONFIG["solver"]["tolerance"], CONFIG["solver"]["max_iterations"])
        out.append(f"Converged: {converged} after {iterations} iterations, last change {change:.2e}")
        out.append("Final values:\n" + value_map(grid, V))
        out.append("Best actions (all ties shown, the first one is used):\n" + policy_map(grid, V))

        policy = make_policy(grid, V)
        for batch in sim["batch_seeds"]:
            results = []
            for i in range(sim["episodes_per_batch"]):
                seed = 1000 * batch + i
                steps, ret, outcome = run_episode(grid, policy, seed, sim["max_steps"])
                results.append((steps, ret, outcome))
                rows.append([name, batch, seed, steps, round(ret, 4), outcome])
            n = len(results)
            out.append(f"batch {batch}: goals {sum(o == 'goal' for _, _, o in results)}/{n}, "
                       f"holes {sum(o == 'hole' for _, _, o in results)}/{n}, "
                       f"time limits {sum(o == 'time limit' for _, _, o in results)}/{n}, "
                       f"mean steps {sum(s for s, _, _ in results) / n:.2f}, "
                       f"mean discounted return {sum(r for _, r, _ in results) / n:.3f}")
        start = find_start(grid)
        out.append(f"V(start) = {V[start]:.3f}")
        out.append("")

    text = "\n".join(out)
    print(text)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "part_c_results.txt").write_text(text, encoding="utf-8")
    with open(ROOT / "outputs" / "part_c_episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["map", "batch_seed", "episode_seed", "steps", "discounted_return", "outcome"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
