import base64
import random
import streamlit as st
import streamlit.components.v1 as components
from mdp_engine import (ARROW, LAYOUTS, GridMDP, constant_policy, greedy_policy, grid_svg, html_player, pi_stages,
                        q_values, run_episode, run_many, sample, value_iteration)
from mdp_tree import acts_fn, build, evaluate, layers, layers_svg, sub, tree_size, tree_svg, zoom_html

st.set_page_config(page_title="MDP Gridworld", page_icon="🧭", layout="wide")
st.title("MDP Gridworld")

LIVING = [-2.0, -1.0, -0.4, -0.1, -0.04, -0.03, -0.01, 0.0, 0.01, 0.1, 1.0]


def controls(prefix, cols):
    """Layout + noise / living reward / discount. Each layout keeps its own defaults (keys include the layout)."""
    name = cols[0].selectbox("Layout", list(LAYOUTS), key=f"{prefix}_layout")
    L = LAYOUTS[name]
    noise = cols[1].slider("Noise", 0.0, 0.5, L['noise'], 0.05, key=f"{prefix}_noise_{name}",
                           help="Probability of slipping sideways: noise/2 to each side. 0.2 → 80% as chosen.")
    living = cols[2].select_slider("Living reward R", LIVING, L['living'], key=f"{prefix}_R_{name}",
                                   help="Reward for every move (not for exiting).")
    gamma = cols[3].slider("Discount γ", 0.0, 1.0, L['gamma'], 0.01, key=f"{prefix}_g_{name}")
    return GridMDP(name, noise, living, gamma)


@st.cache_data
def solve(name, noise, living, gamma):
    m = GridMDP(name, noise, living, gamma)
    Vs = value_iteration(m)
    return Vs, greedy_policy(m, Vs[-1])


def cell_size(m):
    return max(50, min(110, 720 // m.w))


def show_svg(svg):
    # st.html strips <svg>, so the board goes in as an image
    b64 = base64.b64encode(svg.encode()).decode()
    st.html(f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:min(100%,760px)">')


def cell_name(m, c):
    return f"{c}" + (f" exit {m.exits[c]:+g}" if c in m.exits else (f" {m.names[c]}" if m.names.get(c, " ").strip() else ""))


def breakdown(m, ex, vname, mark=None):
    """Markdown table of one Bellman step: per action, the terms p·(R + γ·V(s′)) and Q; the best (or `mark`) in bold."""
    g = f"{m.gamma:g}"
    num = lambda v: f"({v:.2f})" if v < 0 else f"{v:.2f}"
    best = max(q for _, _, q in ex)
    rows = ["| Action | Σ p · (R + γ · " + vname + "(s′)) | Q |", "|---|---|---|"]
    for a, terms, q in ex:
        t = " + ".join(f"{p:g}·({r:g} + {g}·{num(v)})" if n else f"{p:g}·({r:+g})" for p, n, r, v in terms)
        on = (a == mark) if mark else abs(q - best) < 1e-9
        b = "**" if on else ""
        rows.append(f"| {b}{ARROW[a]}{b} | {t} | {b}{q:.2f}{b} |")
    return "\n".join(rows)


def reads(ex):
    return sorted({n for _, terms, _ in ex for _, n, _, _ in terms if n})


tab_grid, tab_vi, tab_tree, tab_pi, tab_about = st.tabs(["Gridworld", "Value iteration", "Tree", "Policy iteration",
                                                        "About"])


def bump(key, d, hi):
    st.session_state[key] = max(0, min(hi, st.session_state[key] + d))


def stepper(col_slider, col_prev, col_next, label, key, hi, prefix):
    """Slider with ◀ ▶; the value resets to 0 when the key changes (new tree, new k ...)."""
    if key not in st.session_state or st.session_state[key] > hi:
        st.session_state[key] = 0
    v = col_slider.slider(label, 0, hi, key=key) if hi else 0
    col_prev.button("◀", disabled=v == 0, on_click=bump, args=(key, -1, hi), key=f"{prefix}_prev")
    col_next.button("▶", disabled=v >= hi, on_click=bump, args=(key, 1, hi), key=f"{prefix}_next")
    return v

# ---------------- Gridworld ----------------
with tab_grid:
    st.subheader("Gridworld: noisy moves, living reward, optimal policy")
    m = controls("g", st.columns(4))
    Vs, pol = solve(m.name, m.noise, m.living, m.gamma)
    o = st.columns([1.4, 1, 1, 1])
    mode = o[0].radio("Mode", ["Drive by hand", "Follow the optimal policy"], horizontal=True)
    show_pol = o[1].checkbox("Show optimal policy", False)
    show_val = o[2].checkbox("Show V*", False)
    seed = o[3].number_input("Seed", 0, 9999, 0, help="Same seed = same slips.")
    V = Vs[-1] if show_val else None
    P = pol if show_pol else None
    cell = cell_size(m)

    if mode == "Drive by hand":
        key = (m.name, m.noise, m.living, m.gamma, int(seed))
        S = st.session_state
        if S.get("drive_key") != key:
            S.drive_key, S.drive = key, dict(s=m.start, ret=0.0, t=0, log=[], done=None, rng=random.Random(seed))
        D = S.drive
        left, right = st.columns([3, 2])
        with right:
            b = st.columns(5)
            pressed = None
            for i, a in enumerate("WNSE"):
                if b[i].button(ARROW[a], key=f"mv{a}", disabled=D['done'] is not None, width="stretch"):
                    pressed = a
            if b[4].button("Exit", disabled=D['done'] is not None or D['s'] not in m.exits, width="stretch"):
                pressed = 'X'
            if pressed and pressed not in m.actions(D['s']):
                st.warning("In an exit cell the only action is Exit." if D['s'] in m.exits else
                           "This layout only moves West / East.")
                pressed = None
            if pressed:
                d, n = sample(m, D['s'], pressed, D['rng'])
                r = m.reward(D['s'], pressed)
                D['ret'] += m.gamma ** D['t'] * r
                D['log'].append(dict(t=D['t'] + 1, chose=ARROW[pressed],
                                     moved=("exit" if n is None else ARROW[d] + (" (bump)" if n == D['s'] else "")),
                                     reward=r, ret=round(D['ret'], 3)))
                D['last'] = (pressed, d if n else None, D['s'])
                D['t'] += 1
                if n is None:
                    D['done'] = m.exits[D['s']]
                else:
                    D['s'] = n
            if st.button("⟲ Restart"):
                S.drive_key = None
                st.rerun()
            c1, c2 = st.columns(2)
            c1.metric("Steps", D['t'])
            c2.metric("Discounted return", f"{D['ret']:.2f}")
            if D['done'] is not None:
                (st.success if D['done'] > 0 else st.error)(f"Exited with {D['done']:+g} after {D['t']} steps.")
            if D['log']:
                st.dataframe(D['log'][::-1], hide_index=True, height=260)
        with left:
            a, d, prev = D.get('last', (None, None, None))
            show_svg(grid_svg(m, cell, V=V, pol=P, agent=D['s'] if D['done'] is None else None, prev=prev,
                              intended=a if a != 'X' and a != d else None, actual=d))
            st.caption("Grey dashed arrow = what you chose, orange arrow = what happened. Every move pays R, the "
                       "exit pays its number. Return = Σ γᵗ·rₜ.")
    else:
        frames, end = run_episode(m, pol, seed=int(seed))
        components.html(html_player(m, frames, cell, pol=P, V=V, speed=3), height=cell * m.h + 90)
        st.caption(f"Seed {seed}: " + (f"exited with {end:+g}" if end is not None else "timeout (100 steps)") +
                   f" after {sum(f['a'] != 'X' for f in frames)} moves, return {frames[-1]['ret']:.2f}.")

    g1, g2 = st.columns([1, 3])
    n_eps = g1.number_input("Episodes (seeds 0 … n−1)", 5, 1000, 20)
    if g1.button("Run many episodes (optimal policy)"):
        r = run_many(m, pol, int(n_eps))
        ends = " · ".join(f"**{'timeout' if k is None else f'{k:+g}'}:** {v}/{int(n_eps)}"
                          for k, v in sorted(r['ends'].items(), key=lambda kv: -(kv[0] or -1e9)))
        g2.write(f"{ends} · **average return:** {r['avg_return']:.2f} · **average moves:** {r['avg_steps']:.1f}")

    with st.expander("Suggested lecture demos (lesson 6)"):
        st.markdown("""
| Demo | Layout | Noise | R | γ | What to show |
|---|---|---|---|---|---|
| Noisy moves | book | 0.2 | 0 | 0.9 | Drive by hand: "up" sometimes slips sideways |
| Deterministic vs stochastic | book | 0 vs 0.2 | 0 | 0.9 | Same buttons, different outcomes |
| Living reward changes the policy | book | 0.2 | −0.01, −0.03, −0.4, −2.0 | 1 | Show optimal policy, move R |
| Safe path vs fast path | bridge | 0 vs 0.2 | 0 | 0.9 | Cross the bridge only without noise |
| Discount quiz | row | 0 | 0 | 1, 0.1, 0.3, 0.33 | Show optimal policy: where does d go? |
""")

# ---------------- Value iteration ----------------
with tab_vi:
    st.subheader("Value iteration: V_k step by step")
    m = controls("v", st.columns(4))
    Vs, final_pol = solve(m.name, m.noise, m.living, m.gamma)
    K = min(len(Vs) - 1, 100)
    key = f"vi_k_{m.name}"
    if key not in st.session_state or st.session_state[key] > K:
        st.session_state[key] = 0
    o = st.columns([3, 1, 1, 1.2, 1.2])
    k = o[0].slider("k (steps left)", 0, K, key=key)
    o[1].button("◀", disabled=k == 0, on_click=lambda key=key, k=k: st.session_state.update({key: k - 1}), key="vi_prev")
    o[2].button("▶", disabled=k >= K, on_click=lambda key=key, k=k: st.session_state.update({key: k + 1}), key="vi_next")
    view = o[3].radio("Show", ["V", "Q"], horizontal=True)
    show_pol = o[4].checkbox("Policy from V_k", True, help="V view only (in the Q view the best action is orange).")
    Vk = Vs[k]
    pol_k = greedy_policy(m, Vk)
    left, right = st.columns([3, 1])
    with left:
        show_svg(grid_svg(m, cell_size(m), V=Vk if view == "V" else None,
                          Q=q_values(m, Vk) if view == "Q" else None,
                          pol=pol_k if show_pol and k and view == "V" else None))
        st.caption("V_k(s) = the best expected (discounted) return when only k steps are left. "
                   "V_{k+1}(s) = max over actions of Σ T·(R + γ·V_k(s′)). In the Q view each cell shows one value "
                   "per action; the best one is orange.")
    with right:
        st.metric("k", k)
        if k:
            st.metric("Largest change from V_{k−1}", f"{max(abs(Vk[s] - Vs[k - 1][s]) for s in m.cells):.4f}")
        stable = next(i for i in range(len(Vs)) if all(greedy_policy(m, Vs[j]) == final_pol
                                                       for j in range(i, len(Vs))))
        st.metric("Policy final from k =", stable)
        st.metric("Values converged at k =", len(Vs) - 1, help="Largest change below 10⁻⁹.")
        if len(Vs) - 1 > K:
            st.caption(f"The slider stops at k = {K}.")

    if k:
        st.markdown(f"#### Inside round {k}: how each V{sub(k)}(s) is computed")
        order = m.cells
        ckey = f"vi_cell_{m.name}_{m.noise}_{m.living}_{m.gamma}_{k}"
        c1, c2, c3, c4 = st.columns([3, .5, .5, 2])
        j = stepper(c1, c2, c3, "Cells computed (engine order: row by row from the bottom-left)", ckey, len(order),
                    "vi_cell")

        def jump(ckey, jkey):
            st.session_state[ckey] = order.index(st.session_state[jkey]) + 1
        jkey = f"vi_jump_{ckey}_{j}"  # new key per position, so the box follows ◀ ▶
        c4.selectbox("Jump to cell", order, index=max(j - 1, 0), format_func=lambda c: cell_name(m, c),
                     key=jkey, on_change=jump, args=(ckey, jkey))
        cur = order[j - 1] if j else None
        ex = m.explain(Vs[k - 1], cur) if cur else None
        small = max(45, cell_size(m) * 3 // 5)
        b1, bt, b2 = st.columns([1, 1.3, 1])
        with b1:
            st.markdown(f"**Read only: V{sub(k - 1)}** (light-blue frames: the cells this step reads)")
            show_svg(grid_svg(m, small, V=Vs[k - 1], read=reads(ex) if ex else (), uid='vir'))
        with b2:
            st.markdown(f"**Written: V{sub(k)}** (orange: the cell just computed)")
            show_svg(grid_svg(m, small, V={c: Vk[c] for c in order[:j]}, changed=[cur] if cur else (), uid='viw'))
        with bt:
            if cur:
                st.markdown(f"**{cell_name(m, cur)}**, V{sub(k)} = max over actions:")
                st.markdown(breakdown(m, ex, f"V{sub(k - 1)}"))
                st.markdown(f"V{sub(k)}{cur} = max = **{Vk[cur]:.2f}**")
            else:
                st.caption("▶ computes the first cell.")
        st.caption(f"The order does not matter: every cell reads only V{sub(k - 1)} (left) and writes to a new "
                   f"table (right), so any order gives the same V{sub(k)}.")

# ---------------- Tree ----------------
MAX_NODES = 2500  # ponytail: fixed cap; the full book tree at k = 3 has ~1,100 state nodes, at k = 4 ~10,000
with tab_tree:
    st.subheader("The tree behind V_k: expectimax, then folded into layers")
    m = controls("t", st.columns(4))
    cells = [c for c in m.cells]
    name = lambda c: f"{c} " + (f"exit {m.exits[c]:+g}" if c in m.exits else m.names.get(c, "start" if c == m.start else ""))
    o = st.columns([1.3, 2.2, .5, .5, 1.3, 1.6])
    root = o[0].selectbox("Root cell (col, row)", cells, format_func=name, key=f"t_root_{m.name}",
                          index=cells.index((2, 2)) if m.name == "book" else cells.index(m.start),
                          help="Row 0 is the bottom row.")
    k = stepper(o[1], o[2], o[3], "k (steps left = depth)", f"t_k_{m.name}", 8, "tk")
    view = o[4].radio("View", ["Tree", "Layers"], horizontal=True, key="t_view",
                      help=f"Tree: the full expectimax tree (up to {MAX_NODES:,} state nodes). "
                           "Layers: the same numbers, one board per V_j.")
    ACTS = {"All actions (max)": None, "Only π*(s)": "opt", "Only East": "E", "Only North": "N"}
    which = o[5].selectbox("Actions at a state node", list(ACTS), key="t_acts",
                           help="All = Value iteration. One action per state = policy evaluation (no max).")
    pol = (None if ACTS[which] is None else solve(m.name, m.noise, m.living, m.gamma)[1] if ACTS[which] == "opt"
           else {s: ("X" if s in m.exits else ACTS[which]) for s in m.cells})
    acts = acts_fn(m, pol)
    n_tree = tree_size(m, root, k, acts)
    Vs = layers(m, k, acts)
    left, right = st.columns([4, 1])
    if view == "Tree":
        with left:
            if n_tree > MAX_NODES:
                st.info(f"With k = {k} the tree has {n_tree:,} state nodes: too many to draw (the limit is "
                        f"{MAX_NODES:,}). "
                        f"Switch to **Layers**: the same values from {k} × {len(m.cells)} updates.")
            else:
                s1, s2, s3, s4 = st.columns([2.2, .5, .5, 1.8])
                step = stepper(s1, s2, s3, "Build step", f"t_step_{m.name}_{root}_{k}_{which}_{m.noise}_{m.living}_{m.gamma}",
                               2 * k + 1, "ts")
                hl = s4.selectbox("Mark every copy of cell", [None] + cells, key=f"t_hl_{m.name}",
                                  format_func=lambda c: "—" if c is None else name(c))
                T = build(m, root, k, acts)
                evaluate(m, T)
                components.html(zoom_html(tree_svg(m, T, step, hl)), height=690)
                lvl = ("nothing yet" if step == 0 else "the leaves: V₀ = 0" if step == 1 else
                       f"chance nodes with {step // 2} step{'s' if step // 2 > 1 else ''} left: average over outcomes" if step % 2 == 0 else
                       f"state nodes V{sub(step // 2)}: max over the chance nodes" if acts == m.actions else
                       f"state nodes V{sub(step // 2)}: the policy's only action (no max)")
                st.caption(f"**Step {step} / {2 * k + 1}: {lvl}.** Blue board = state node (the green dot is the "
                           "agent). Circle = chance node (action), with its value in yellow. Dashed edges carry the "
                           "probability. Q = Σ p·(R + γ·V); an exit pays R and ends (end = 0). Orange = the best action.")
                if hl is not None:
                    copies = []
                    stack = [T]
                    while stack:
                        n = stack.pop()
                        copies += [n] if n['kind'] == 'S' and n['s'] == hl else []
                        stack += [c[1] if isinstance(c, tuple) else c for c in n.get('ch', [])]
                    st.caption(f"Cell {hl} appears **{len(copies)}** times in this tree, and each copy is computed "
                               "again. Value iteration computes it once per layer.")
    else:
        with left:
            s1, s2, s3, _ = st.columns([2.2, .5, .5, 1.8])
            step = stepper(s1, s2, s3, "Layer computed", f"t_layer_{m.name}_{root}_{k}_{which}_{m.noise}_{m.living}_{m.gamma}",
                           k, "tl")
            components.html(zoom_html(layers_svg(m, Vs, step, root, acts, cell=cell_size(m) * 2 // 3), height=690),
                            height=740)
            st.caption("One board per V_j, V₀ at the bottom. Each ▶ computes one whole layer from the layer below: "
                       f"Bellman update on every cell. The frames show the update for {root} on the newest layer: it "
                       "reads only its next cells one layer down (orange = the best action, with its probabilities; dashed = "
                       "the other actions). The tree repeats the same "
                       "cells again and again; the layers compute each cell once per k.")
    with right:
        st.metric(f"V{sub(k)}{root}", f"{Vs[k][root]:.2f}")
        st.metric("Tree: state nodes", f"{n_tree:,}")
        st.metric("Layers: Bellman updates", f"{k * len(m.cells):,}", help=f"{k} layers × {len(m.cells)} cells")
        if which != "All actions (max)":
            st.caption("One action per state: the tree is much narrower, and each update needs no max "
                       "(policy evaluation).")

with tab_pi:
    st.subheader("Policy iteration: evaluate, improve, repeat")
    m = controls("p", st.columns(4))
    START = {"Always East": "E", "Always North": "N", "Always West": "W", "Always South": "S", "Optimal": None}
    o = st.columns([2, 1, 1, 1])
    start = o[0].selectbox("Start policy", list(START), key="pi_start")
    pol0 = constant_policy(m, START[start]) if START[start] else solve(m.name, m.noise, m.living, m.gamma)[1]
    stages = pi_stages(m, pol0)
    key = f"pi_i_{m.name}_{m.noise}_{m.living}_{m.gamma}_{start}"
    st.session_state.setdefault(key, 0)
    i = min(st.session_state[key], len(stages) - 1)
    nxt = stages[i + 1]['kind'] if i + 1 < len(stages) else None
    o[1].button("◀ Back", disabled=i == 0, on_click=lambda key=key, i=i: st.session_state.update({key: i - 1}), key="pi_prev")
    o[2].button({"eval": "Evaluate ▶", "improve": "Improve ▶"}.get(nxt, "Done ✓"), disabled=nxt is None,
                on_click=lambda key=key, i=i: st.session_state.update({key: i + 1}), key="pi_next")
    o[3].button("⟲ Reset", on_click=lambda key=key: st.session_state.update({key: 0}), key="pi_reset")
    S = stages[i]
    pcell = None
    if S['kind'] != 'start':
        free = [c for c in m.cells if c not in m.exits]
        pcell = st.columns([1, 2])[0].selectbox("Show the computation for cell", free, index=free.index(m.start) if m.start in free else 0,
                             format_func=lambda c: cell_name(m, c), key=f"pi_cell_{m.name}")
        old = stages[i - 1]['pol'][pcell] if S['kind'] == 'improve' else S['pol'][pcell]
        pex = m.explain(S['V'], pcell, None if S['kind'] == 'improve' else [S['pol'][pcell]])
    left, mid, right = st.columns([2.2, 1.6, 1])
    with left:
        show_svg(grid_svg(m, cell_size(m), V=S['V'], pol=S['pol'], changed=S['changed'], uid='pi',
                          read=reads(pex) if pcell else ()))
        st.caption({"start": "The start policy: one arrow per cell, no values yet.",
                    "eval": "Policy evaluation: V^π(s) = Σ T·(R + γ·V^π(s′)) with the action π(s) only (no max). "
                            "Orange frames: values that changed by more than 0.005.",
                    "improve": "Policy improvement (policy extraction): in every cell, the action with the best "
                               "one-step look-ahead on V^π. Orange frames: arrows that changed."}[S['kind']])
    with mid:
        if S['kind'] == 'eval':
            st.markdown(f"**{cell_name(m, pcell)}**: only the policy's action {ARROW[old]}, no max. V^π is the final "
                        f"result of {S['sweeps']} sweeps, so the step gives back its own value:")
            st.markdown(breakdown(m, pex, "V^π", mark=old))
            st.markdown(f"V^π{pcell} = **{S['V'][pcell]:.2f}** (light-blue frames: the cells it reads)")
        elif S['kind'] == 'improve':
            new = S['pol'][pcell]
            st.markdown(f"**{cell_name(m, pcell)}**: every action, one step on V^π, then the best one:")
            st.markdown(breakdown(m, pex, "V^π", mark=new))
            ties = sum(1 for _, _, q in pex if abs(q - max(x for _, _, x in pex)) < 1e-9)
            st.markdown(f"Was {ARROW[old]}, now **{ARROW[new]}** (changed)." if new != old else
                        f"Stays **{ARROW[new]}**" + (": a tie keeps the current action." if ties > 1 else "."))
    with right:
        rounds = sum(1 for x in stages[1:i + 1] if x['kind'] == 'improve' and x['changed'])
        st.metric("Step", {"start": "Start", "eval": "Evaluate", "improve": "Improve"}[S['kind']])
        st.metric("Improvements that changed arrows", rounds)
        if S['kind'] == 'eval':
            st.metric("Evaluation sweeps", S['sweeps'], help="Sweeps without max until the largest change < 10⁻⁹.")
        if S['kind'] != 'start':
            st.metric("Cells changed", len(S['changed']))
        if nxt is None:
            st.success("No arrow changed: this is the optimal policy π*.")

with tab_about:
    st.subheader("What this demonstrates")
    st.markdown("""
- **Gridworld (lesson 6):** a Markov decision process on a grid. Moves are noisy (with noise 0.2, a move goes
  where you chose 80% of the time and slips to each side 10%), every move pays the living reward R, and exit cells
  pay their number and end the episode. Drive by hand, or watch the optimal policy act, and see how R, the noise
  and γ change the optimal policy.
- **Value iteration (lesson 7):** V_k for k = 0, 1, 2, … with the greedy policy and the Q-values. The policy usually
  becomes final long before the values converge. "Inside round k" fills V_k one cell at a time from V_{k−1}, with
  the full calculation per action; any order gives the same V_k, because every cell reads only V_{k−1}.
- **Tree (lesson 7):** the expectimax tree whose root value is V_k(s), built level by level (leaves, averages,
  max), and the same numbers folded into one board per V_j. The tree grows about ×10 per level, yet every level has
  at most one node per cell: Value iteration computes each cell once per layer.
- **Policy iteration (lesson 7):** start from a fixed policy, evaluate it (no max), improve it with a one-step
  look-ahead, and repeat until no arrow changes. Choosing "Optimal" as the start shows policy evaluation of π*,
  and the calculation for any cell: the policy's action only (Evaluate), or every action and the best one (Improve).

The engine is `mdp_engine.py` (standard library only); the Colab exercise HM_06 uses the same file.
Self-check: `python3 mdp_engine.py`. This is a clean-room educational implementation with its own look.
""")
