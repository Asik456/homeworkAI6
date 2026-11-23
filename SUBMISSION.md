# HW6 submission

## AI-use declaration

I used Claude Code (Anthropic) to help me write the two programs (`frozenlake.py`, `solver.py`), check my Part A numbers and write this file from my own runs. I did the Part A calculations by hand.

## Setup and reproduction

Python 3.12.15, gymnasium 1.4.0, numpy 2.5.4 (`requirements.txt`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python frozenlake.py     # Part B -> outputs/part_b_*.{csv,md,txt}
python solver.py         # Part C -> outputs/part_c_results.txt, outputs/part_c_episodes.csv
```

## Part A — Paper calculations

Photo of my paper work: [part_a_paper.jpg](part_a_paper.jpg). The same steps are written below.

**1. The MDP in my own words.**
- **States:** the 9 tiles where the agent can stand: (1,1), (1,2) mud, (1,3), (2,1), (2,3) ice, (3,1) start, (3,2), (3,3), (3,4). G and H are end states. The wall is not a state.
- **Actions:** up, right, down, left.
- **Transitions:** on normal tiles and mud the move always happens. On ice: 0.8 the chosen way, 0.1 each sideways. Hitting a wall or the edge means staying in place.
- **Rewards:** the reward of the tile where the move ends: floor -1, mud -3, goal +10, hole -10.
- **Goal of the agent:** get the biggest total discounted reward (discount 0.9), so reach G fast, avoid H and mud.

**2. V1 (from V0 = 0, so V1 = best immediate reward):**

```text
        1      2      3      4
1     -1     -1     10      G
2     -1      #     -1      H
3     -1     -1     -1     -1
```

Examples: (1,3) right goes into G: +10. (1,2) mud: right to (1,3) gives -1 (staying in mud would give -3). Ice (2,3): left hits the wall: 0.8(-1) + 0.1(-1) + 0.1(-1) = -1, better than up or down: 0.8(-1) + 0.1(-1) + 0.1(H -10) = -1.9.

**3. V2 using only V1** (value of a move = reward + 0.9 × V1(next tile)):

(1,2) mud:
- up: wall/edge → stay in mud: -3 + 0.9(-1) = -3.9
- right: (1,3): -1 + 0.9(10) = **8**
- down: wall → stay: -3 + 0.9(-1) = -3.9
- left: (1,1): -1 + 0.9(-1) = -1.9
- **V2 = 8**

(1,3):
- up: edge → stay: -1 + 0.9(10) = 8
- right: G: +10 + 0 = **10**
- down: ice: -1 + 0.9(-1) = -1.9
- left: mud: -3 + 0.9(-1) = -3.9
- **V2 = 10**

(2,3) ice. First the value of each place I can land: up (1,3) = -1 + 0.9(10) = 8; right H = -10; down (3,3) = -1 + 0.9(-1) = -1.9; left wall → stay on ice = -1 + 0.9(-1) = -1.9.
- up: 0.8(8) + 0.1(left -1.9) + 0.1(right -10) = 6.4 - 0.19 - 1 = **5.21**
- right: 0.8(-10) + 0.1(up 8) + 0.1(down -1.9) = -8 + 0.8 - 0.19 = -7.39
- down: 0.8(-1.9) + 0.1(left -1.9) + 0.1(right -10) = -1.52 - 0.19 - 1 = -2.71
- left: 0.8(-1.9) + 0.1(up 8) + 0.1(down -1.9) = -1.52 + 0.8 - 0.19 = -0.91
- **V2 = 5.21**

**4. Best actions:** (1,2) → right, (1,3) → right, (2,3) → up. No ties at these three tiles.

Two rounds are not enough to prove the final policy: V2 only looks 2 steps ahead. Values far from the goal (like the start) still have not "seen" the +10, and they will keep changing in later rounds. We can only trust the policy when the values stop changing (convergence).

## Part B — FrozenLake

### Fixed policy, written before the experiments

States are numbered 0–15 row by row. Holes: 5, 7, 11, 12. Goal: 15. My path goes down the left side, then right along the bottom: 0 → 4 → 8 → 9 → 13 → 14 → 15. Without slipping this path never enters a hole.

| State | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 10 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Action | down | left | down | left | down | down | right | down | down | right | right |

States off the path (1, 2, 3, 6, 10) send the agent back towards the path or down to the bottom row. The policy is a fixed table in `frozenlake.py` (`FIXED_POLICY`) and did not change between experiments.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | random | 11 | 0 | 50 | 0 | 0% | 8.14 |
| False | random | 22 | 1 | 49 | 0 | 2% | 7.56 |
| False | random | 33 | 0 | 50 | 0 | 0% | 6.88 |
| False | fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| True | random | 11 | 0 | 50 | 0 | 0% | 8.06 |
| True | random | 22 | 0 | 50 | 0 | 0% | 7.60 |
| True | random | 33 | 0 | 50 | 0 | 0% | 7.98 |
| True | fixed | 11 | 1 | 49 | 0 | 2% | 5.22 |
| True | fixed | 22 | 2 | 48 | 0 | 4% | 6.00 |
| True | fixed | 33 | 1 | 49 | 0 | 2% | 5.14 |

Every row is in `outputs/part_b_episodes.csv`. Traces (text rendering) of one failed and one successful slippery episode are in `outputs/part_b_traces.txt`. Time limits never happened: every episode ended in a hole or the goal before 100 steps.

**What slipping changed.** Without slipping, my fixed policy reached the goal 150/150 times in exactly 6 steps, because every move does what I choose. With `is_slippery=True` the agent goes the chosen way only 1/3 of the time (the other 2/3 go sideways, per the FrozenLake docs), so my path next to the holes fails: only 4/150 goals. The random policy almost never succeeds (1/150 without slipping, 0/150 with slipping).

**Consistency.** The batches are very close (fixed, slippery: 1, 2, 1 goals), so the result is stable, but there are small changes from batch to batch because the slips are random.

**Why one success is not enough.** One success can be luck: the random policy reached the goal once (batch 22), but its real success rate is about 1%. We need many episodes to estimate the rate.

**Why this is not training.** The policy is a fixed table I wrote before running. Nothing is learned or updated from the rewards; the runs only measure how good this fixed policy is.

## Part C — Custom solver

### Compare code with paper

`solver.py` prints V1 and the V2 calculations. They match my paper work exactly, so there were no discrepancies:

```text
V1:  . -1   M -1   . 10   G 0
     . -1     #    I -1   H 0
     S -1   . -1   . -1   . -1
V2 at (1,2): up -3.9, right 8.0, down -3.9, left -1.9 -> 8.0
V2 at (1,3): up 8.0, right 10.0, down -1.9, left -3.9 -> 10.0
V2 at (2,3): up 5.21, right -7.39, down -2.71, left -0.91 -> 5.21
```

### Values, policies and convergence

Full output: `outputs/part_c_results.txt`. Tie rule: I compare actions in the order up, right, down, left and take the first one with the biggest value (the map prints all tied actions; at convergence there were no ties).

| Map | Iterations | Last change |
|---|---|---|
| paper map | 12 | 2.17e-07 |
| extended map | 11 | 2.41e-07 |

```text
Paper map values                       Paper map policy
  . 4.20   M 8.00  . 10.00   G             →   →   →   G
  . 2.78      #    I 5.82    H             ↑   #   ↑   H
  S 1.54   . 2.82  . 4.24  . 2.82          →   →   ↑   ←

Extended map values                    Extended map policy
  . 4.20   M 8.00  . 10.00   G             →   →   →   G
  . 2.78      #    I 5.82    H             ↑   #   ↑   H
  S 1.50   D 2.82  . 4.24  W 2.82          ↑   →   ↑   ←
```

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 42 | 8 | 0 | 4.98 | 0.891 |
| paper | 22 | 46 | 4 | 0 | 5.04 | 1.915 |
| paper | 33 | 48 | 2 | 0 | 5.04 | 2.492 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.502 |

Every episode is in `outputs/part_c_episodes.csv`.

### Explain your results

**What changed with sand and water.** Only the start changed. On the paper map the start goes right: S → (3,2) → (3,3) → ice → (1,3) → G, V(S) = 1.54. On the extended map (3,2) is sand, so entering it costs -2 instead of -1. Now the safe way up is better: S → (2,1) → (1,1) → mud → (1,3) → G, V(S) = 1.50. The water at (3,4) did not change anything, because the best path never enters it. The value printed on D and W is the value of standing there; the cost is paid when you enter.

**How the policy handled ice.** At the ice the best action is up: 0.8 it reaches (1,3) next to the goal, 0.1 it stays, but 0.1 it slips right into the hole. That is why the paper map has holes in the simulation (14/150 ≈ 9%). The extended map avoids the ice, so it never falls: 150/150 goals.

**Cost vs risk.** On the paper map, the ice path is cheaper (no mud) but risky; the solver still chose it because its expected value (1.54) is a bit higher than the safe mud path. When sand made the ice path more expensive, the safe path won.

**Simulations vs values.** V(S) is an expected value over infinitely many episodes. One batch of 50 is a finite sample, so the paper map batch means (0.89, 1.92, 2.49) jump around V(S) = 1.54 because of random slips; their average 1.77 is close. On the extended map there is no randomness on the path, so every episode gives exactly 1.502 = V(S).

### Rule checks

From `outputs/part_c_results.txt`:
- Wall: from (3,2) moving up stays at (3,2), because (2,2) is a wall.
- Boundary: from (1,1) moving left stays at (1,1).
- Ice (2,3): probabilities for every action sum to 1.0 (0.8 + 0.1 + 0.1).
- Rewards: entering mud -3, sand -2, water -5, goal +10.
- After termination: G and H stay 0 and no more rewards are added (the simulation stops at G or H; the update skips terminal states).
