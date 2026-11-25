import base64
import random
import streamlit as st
import streamlit.components.v1 as components
from mdp_engine import (ARROW, LAYOUTS, GridMDP, constant_policy, greedy_policy, grid_svg, html_player, pi_stages,
                        q_values, run_episode, run_many, sample, value_iteration)

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


tab_grid, tab_vi, tab_pi, tab_about = st.tabs(["Gridworld", "Value iteration", "Policy iteration", "About"])

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
    left, right = st.columns([3, 1])
    with left:
        show_svg(grid_svg(m, cell_size(m), V=S['V'], pol=S['pol'], changed=S['changed'], uid='pi'))
        st.caption({"start": "The start policy: one arrow per cell, no values yet.",
                    "eval": "Policy evaluation: V^π(s) = Σ T·(R + γ·V^π(s′)) with the action π(s) only (no max). "
                            "Orange frames: values that changed by more than 0.005.",
                    "improve": "Policy improvement (policy extraction): in every cell, the action with the best "
                               "one-step look-ahead on V^π. Orange frames: arrows that changed."}[S['kind']])
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
  becomes final long before the values converge.
- **Policy iteration (lesson 7):** start from a fixed policy, evaluate it (no max), improve it with a one-step
  look-ahead, and repeat until no arrow changes. Choosing "Optimal" as the start shows policy evaluation of π*.

The engine is `mdp_engine.py` (standard library only); the Colab exercise HM_06 uses the same file.
Self-check: `python3 mdp_engine.py`. This is a clean-room educational implementation with its own look.
""")
