# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

Record dependency versions, installation commands and exact run commands.

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

State the action for each nonterminal state and your reasoning.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |

Link episode rows and example traces in outputs/. Explain slipping, variation across batches and why these runs do not train a policy.

## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

