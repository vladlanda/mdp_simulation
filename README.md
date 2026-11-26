# MDP Gridworld

Streamlit app for lessons 6–7 (MDPs): a grid world with noisy moves, living reward and discount, driven by hand
or by the optimal policy, value iteration step by step (V and Q views), the expectimax tree behind V_k (and the same
numbers folded into layers), and policy iteration (evaluate / improve).

`mdp_engine.py` is the engine (standard library only). The Colab exercise HM_06 uses the same code.
Self-check: `python3 mdp_engine.py` (checks the CS188 V\* values, the lesson 6 policies and the discount quiz).
`mdp_tree.py` builds and draws the tree for the Tree tab; self-check `python3 mdp_tree.py` (tree root = V_k for every
cell, k ≤ 3).

Layouts: `book` (the 4x3 world from the lecture), `bridge`, `cliff`, `discount` (CS188 grids), `row`
(the lesson 6 discount quiz, states a–e) and `corridor` (−1 | A | B | +1, the lesson 7 worked example). Noise 0.2 means 80% as chosen and 10% to each side; a bump into a wall
leaves the agent in place. Every move pays R, an exit pays its number.

## Run

```bash
uv sync
uv run streamlit run app.py
```

## Lecture demos (measured, optimal policy, seeds 0–19)

| Demo | Settings | Result |
|---|---|---|
| Deterministic world | book · noise 0 · R 0 · γ 0.9 | 20/20 reach +1 in 5 moves, return 0.59 |
| Noisy world | book · noise 0.2 · R 0 · γ 0.9 | 20/20 reach +1, avg 6.3 moves, avg return 0.52 |
| Living reward −0.01 | book · noise 0.2 · γ 1 | 20/20 reach +1, avg 6.5 moves |
| Living reward −0.03 | book · noise 0.2 · γ 1 | 20/20 reach +1, avg 6.3 moves |
| Living reward −0.4 | book · noise 0.2 · γ 1 | 20/20 reach +1, avg 6.0 moves |
| Living reward −2.0 | book · noise 0.2 · γ 1 | 19/20 end in −1, avg 4.8 moves (along the bottom and into −1: any exit, fast) |
| Bridge without noise | bridge · noise 0 · γ 0.9 | crosses to +10: 20/20, 5 moves |
| Bridge with noise | bridge · noise 0.2 · γ 0.9 | turns back to +1: 17/20 (3/20 fall to −100) |
| Cliff | cliff · noise 0 / 0.2 · γ 0.9 | noise 0: 20/20 to +10 · noise 0.2: 17/20 take the +8 |
| Discount quiz | row · γ 1 / 0.3 / 0.33 / 0.1 | d goes West at γ = 1 and 0.33, East at 0.3 and 0.1 (tie at 1/√10 ≈ 0.316) |
| Value iteration | book · noise 0.2 · R 0 · γ 0.9 | V_2 = 0.72 next to +1 · policy final from k = 10 · values converge (10⁻⁹) at k = 32 |
| Inside round k (lesson 7) | Value iteration tab · book · k 3 · jump to (2, 2) | left of +1: ↑ 0.61, ↓ 0.09, → 0.78, ← 0.06 → V₃ = 0.78 · two boards (read V_{k−1}, write V_k), order does not change V_k |
| V vs Q (lesson 7) | book · noise 0.2 · R 0 · γ 0.9 · k 100 | below the −1: W 0.28, S 0.27, E 0.13, N −0.65 |
| Corridor (lesson 7 worked example) | corridor · Value iteration / Tree tab · root B | V_k for (A, B): (0, 0), (0, 0), (0, 0.72), (0.52, 0.85), (0.71, 0.87) for k = 0–4, converges at k = 17 to (0.771, 0.878) · full tree from B: 17 / 53 / 161 state nodes for k = 2–4 |
| Tree (lesson 7) | Tree tab · book · root (2, 2) | k = 2: V₂ = 0.72 (→ 0.72, ↑ 0.09, ↓ 0.09, ← 0) · state nodes in the full tree: 13 / 115 / 1,063 / 10,075 for k = 1–4 (from the start cell: 11 / 111 / 1,129 / 11,561), never more than 11 different cells per level · only π*: 10 nodes at k = 2 |
| Policy evaluation (lesson 7) | Policy iteration tab, Evaluate | start cell: Always East −0.30, Always North 0.05, π* 0.49 |
| Inside a PI step (lesson 7) | Policy iteration tab · Always East · Evaluate: sweep slider + cell walk; Improve: cell walk | Evaluate: 92 sweeps, read V^π_{j−1} / table (→ only, no max) / write V^π_j; sweep 2 next to the pit −0.72 · Improve at (0, 0): ↑ −0.01, ← −0.24, ↓ −0.28, → −0.30 → changes to ↑ |
| Policy iteration (lesson 7) | Policy iteration tab, start Always East / North | improvements change 6 then 2 cells (North: 5 then 1), then none: π* |

The V\* values for `book` (γ 0.9, noise 0.2) match the CS188 lecture snapshot: 0.64 0.74 0.85 / 0.57 · 0.57 /
0.49 0.43 0.48 0.28.

## Regenerating the Colab exercise

`build_hm06.py` rebuilds `moodle/HM_06.ipynb` and `HM_06_solution.ipynb` from this engine. The student copy has
no transition model, no sampling and no planning code; the optimal policies it needs are precomputed:

```bash
python build_hm06.py mdp_engine.py ../../moodle answers_hm06.json
```

Needs `nbformat` (and `nbclient` + `ipykernel` to execute the notebooks for a check).
`python build_hm06.py mdp_engine.py <dir> --numbers` prints the reference numbers used in the answers.

`build_hm07.py` does the same for HM_07 (lesson 7: Bellman step, value iteration, policy extraction, policy
evaluation, policy iteration, and a bonus race-car MDP). The student copy keeps the transition model (HM_06) and
drops all the planning code:

```bash
python build_hm07.py mdp_engine.py ../../moodle answers_hm07.json
```
