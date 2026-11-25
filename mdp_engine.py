"""Grid-world MDP engine for the lesson 6-8 demos and the HM_06 exercise (standard library only).

Cells are (col, row) with row 0 at the BOTTOM (as in CS188). An exit cell holds a number: its only action is
'X' (exit), which pays that number and ends the episode. Moves N/S/E/W succeed with probability 1 - noise and slip
to each perpendicular side with noise / 2; a move into a wall or off the board leaves the agent in place.
Every move pays the living reward R. Self-check: python3 mdp_engine.py
"""
import json
import random

DIRS = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
SIDE = {'N': 'WE', 'S': 'EW', 'E': 'NS', 'W': 'SN'}
ARROW = {'N': '↑', 'S': '↓', 'E': '→', 'W': '←', 'X': 'exit'}

# rows top to bottom; '.' empty, '#' wall, 'S' start, a number = exit with that reward
LAYOUTS = {
    'book': dict(grid="""
        .  .  .  +1
        .  #  .  -1
        S  .  .  .""", noise=0.2, living=0.0, gamma=0.9),
    'bridge': dict(grid="""
        #  -100 -100 -100 -100 -100  #
        1  S    .    .    .    .     10
        #  -100 -100 -100 -100 -100  #""", noise=0.2, living=0.0, gamma=0.9),
    'cliff': dict(grid="""
        .    .    .    .    .
        8    S    .    .    10
        -100 -100 -100 -100 -100""", noise=0.2, living=0.0, gamma=0.9),
    'discount': dict(grid="""
        .   .   .   .   .
        .   #   .   .   .
        .   #   1   #   10
        S   .   .   .   .
        -10 -10 -10 -10 -10""", noise=0.2, living=0.0, gamma=0.9),
    # lesson 6 discount quiz: states a..e, deterministic; at a and e the agent may also exit (or keep walking)
    'row': dict(grid="""
        10  .  .  .  1""", noise=0.0, living=0.0, gamma=1.0, walk_from_exits=True, start=(3, 0),
                names='abcde', moves='WE'),  # ponytail: West first, so the gamma = 1 ties read 'go West' as in the quiz
    # lesson 7 worked example: -1 | A | B | +1; a move succeeds w.p. 0.8, else the agent slips into a side wall and stays
    'corridor': dict(grid="""
        -1  .  S  +1""", noise=0.2, living=0.0, gamma=0.9, names=' AB ', moves='WE'),  # start at B: the tree tab's root
}


class GridMDP:
    def __init__(self, layout='book', noise=None, living=None, gamma=None):
        L = LAYOUTS[layout] if isinstance(layout, str) else layout
        self.name = layout if isinstance(layout, str) else 'custom'
        rows = [r.split() for r in L['grid'].strip('\n').splitlines() if r.strip()]
        self.h, self.w = len(rows), len(rows[0])
        self.walls, self.exits, self.start = set(), {}, L.get('start')
        for i, row in enumerate(rows):
            for x, t in enumerate(row):
                c = (x, self.h - 1 - i)
                if t == '#':
                    self.walls.add(c)
                elif t == 'S':
                    self.start = self.start or c
                elif t != '.':
                    self.exits[c] = float(t)
        self.cells = [(x, y) for y in range(self.h) for x in range(self.w) if (x, y) not in self.walls]
        self.start = self.start or self.cells[0]
        self.walk_from_exits = L.get('walk_from_exits', False)
        self.moves = list(L.get('moves', DIRS))
        self.names = dict(zip(self.cells, L['names'])) if 'names' in L else {}
        self.noise = L['noise'] if noise is None else noise
        self.living = L['living'] if living is None else living
        self.gamma = L['gamma'] if gamma is None else gamma

    def actions(self, s):
        if s in self.exits:
            return ['X'] + (self.moves if self.walk_from_exits else [])
        return self.moves

    def step_to(self, s, d):
        t = (s[0] + DIRS[d][0], s[1] + DIRS[d][1])
        return t if t in self.cells else s

    def outcomes(self, s, a):
        """[(probability, direction actually taken, next state)]; next state None = episode over."""
        if a == 'X':
            return [(1.0, 'X', None)]
        out = [(1 - self.noise, a)] + [(self.noise / 2, d) for d in SIDE[a]]
        return [(p, d, self.step_to(s, d)) for p, d in out if p > 0]

    def reward(self, s, a):
        return self.exits[s] if a == 'X' else self.living

    def q(self, V, s, a):
        r = self.reward(s, a)
        return sum(p * (r + (self.gamma * V[n] if n else 0.0)) for p, _, n in self.outcomes(s, a))

    def explain(self, V, s, actions=None):
        """One Bellman step at s, term by term: [(a, [(p, next state or None, R, V[next] or None)], Q)].
        Slips into the same cell are merged into one term."""
        out = []
        for a in actions or self.actions(s):
            r, terms = self.reward(s, a), {}
            for p, _, n in self.outcomes(s, a):
                terms[n] = terms.get(n, 0.0) + p
            out.append((a, [(p, n, r, V[n] if n else None) for n, p in terms.items()], self.q(V, s, a)))
        return out


# ---------------- planning ----------------
def value_iteration(mdp, k=None, tol=1e-9, k_max=1000):
    """[V_0, V_1, ..., V_k]. With k=None: until the largest change is below tol (at most k_max steps)."""
    Vs = [{s: 0.0 for s in mdp.cells}]
    for _ in range(k if k is not None else k_max):
        V = Vs[-1]
        Vs.append({s: max(mdp.q(V, s, a) for a in mdp.actions(s)) for s in mdp.cells})
        if k is None and max(abs(Vs[-1][s] - V[s]) for s in mdp.cells) < tol:
            break
    return Vs


def q_values(mdp, V):
    return {s: {a: mdp.q(V, s, a) for a in mdp.actions(s)} for s in mdp.cells}


def greedy_policy(mdp, V, eps=1e-9):
    """Best action per cell; ties go to the first of X, N, S, E, W."""
    pol = {}
    for s in mdp.cells:
        Q = {a: mdp.q(V, s, a) for a in mdp.actions(s)}
        best = max(Q.values())
        pol[s] = next(a for a in Q if Q[a] >= best - eps)
    return pol


def policy_evaluation(mdp, pol, k=None, tol=1e-9, k_max=1000):
    """[V^pi_0, ..., V^pi_k] for a fixed policy (iterative evaluation)."""
    Vs = [{s: 0.0 for s in mdp.cells}]
    for _ in range(k if k is not None else k_max):
        V = Vs[-1]
        Vs.append({s: mdp.q(V, s, pol[s]) for s in mdp.cells})
        if k is None and max(abs(Vs[-1][s] - V[s]) for s in mdp.cells) < tol:
            break
    return Vs


def policy_iteration(mdp, pol=None):
    """[(policy, its values)] until the policy stops changing. Starts from 'always N' (exit where forced)."""
    pol = pol or {s: mdp.actions(s)[0] if s in mdp.exits else mdp.moves[0] for s in mdp.cells}
    steps = []
    while True:
        V = policy_evaluation(mdp, pol)[-1]
        steps.append((pol, V))
        new = greedy_policy(mdp, V)
        if all(abs(mdp.q(V, s, new[s]) - mdp.q(V, s, pol[s])) < 1e-9 for s in mdp.cells):
            return steps
        pol = new


def pi_stages(mdp, pol, tol=0.005):
    """Policy iteration as a list of stages for display: ('start', pol), then alternating ('eval', pol, V, changed
    values, sweeps) and ('improve', new pol, V, changed arrows), ending with an improve step that changes nothing.
    A tie keeps the current action, so only real improvements flip an arrow."""
    st = [dict(kind='start', pol=pol, V=None, changed=set())]
    while True:
        Vs = policy_evaluation(mdp, pol)
        V, prevV = Vs[-1], st[-1]['V']
        st.append(dict(kind='eval', pol=pol, V=V, sweeps=len(Vs) - 1,
                       changed={s for s in mdp.cells if prevV is not None and abs(V[s] - prevV[s]) > tol}))
        best = greedy_policy(mdp, V)
        new = {s: pol[s] if mdp.q(V, s, best[s]) - mdp.q(V, s, pol[s]) < 1e-9 else best[s] for s in mdp.cells}
        ch = {s for s in mdp.cells if new[s] != pol[s]}
        st.append(dict(kind='improve', pol=new, V=V, changed=ch))
        if not ch:
            return st
        pol = new


def constant_policy(mdp, a):
    return {s: mdp.actions(s)[0] if s in mdp.exits else a for s in mdp.cells}


# ---------------- episodes ----------------
def sample(mdp, s, a, rng):
    """One transition: (direction actually taken, next state)."""
    r, acc = rng.random(), 0.0
    outs = mdp.outcomes(s, a)
    for p, d, n in outs:
        acc += p
        if r < acc:
            return d, n
    return outs[-1][1:]


def run_episode(mdp, pol, seed=0, start=None, max_steps=100):
    """Follow a policy. Returns frames: dict(s, a = intended, d = actually taken, r, ret = discounted return so
    far, t) and the result: the exit value reached, or None on timeout."""
    rng = random.Random(seed)
    s, ret, frames = start or mdp.start, 0.0, []
    for t in range(max_steps):
        a = pol[s]
        d, n = sample(mdp, s, a, rng)
        r = mdp.reward(s, a)
        ret += mdp.gamma ** t * r
        frames.append(dict(t=t, s=s, a=a, d=d, n=n, r=r, ret=ret))
        if n is None:
            return frames, mdp.exits[s]
        s = n
    return frames, None


def run_many(mdp, pol, n=20, start=None, max_steps=100):
    """Episodes on seeds 0..n-1: {exit value or None: count}, average return, average steps."""
    ends, rets, steps = {}, [], []
    for i in range(n):
        fr, end = run_episode(mdp, pol, seed=i, start=start, max_steps=max_steps)
        ends[end] = ends.get(end, 0) + 1
        rets.append(fr[-1]['ret'])
        steps.append(sum(f['a'] != 'X' for f in fr))
    return dict(ends=ends, avg_return=sum(rets) / n, avg_steps=sum(steps) / n)


# ---------------- drawing (SVG, own look) ----------------
BG, WALL, LINE, TXT, ORANGE, AGENT, READ = '#0b1020', '#2b6cb0', '#334155', '#e2e8f0', '#f59e0b', '#22c55e', '#38bdf8'


def fmt(v):
    return f'{v:.2f}'.replace('-0.00', '0.00')


def _shade(v, vmax):
    """Green for positive, red for negative, stronger with |v| (as in the CS188 value displays)."""
    a = min(1.0, abs(v) / vmax) if vmax else 0
    base = (34, 197, 94) if v > 0 else (239, 68, 68)
    mix = lambda c, b: round(b + (c - b) * a * 0.75)
    return '#%02x%02x%02x' % tuple(mix(c, b) for c, b in zip(base, (15, 23, 42)))


def grid_svg(mdp, cell=90, V=None, Q=None, pol=None, agent=None, prev=None, intended=None, actual=None, trail=(),
             show_names=True, uid='g', changed=(), read=()):
    """Board as an SVG string. V: values per cell, Q: {s: {a: q}} (drawn as four triangles), pol: arrows,
    agent: cell of the round agent; prev: the cell it came from, intended / actual: the last move's chosen and
    actual directions (grey dashed / orange arrow out of prev; a bump into a wall is a short arrow back).
    V may cover only some cells (the rest are drawn empty); changed: orange frames; read: light-blue dashed frames."""
    W, H = mdp.w * cell, mdp.h * cell
    X = lambda c: c[0] * cell
    Y = lambda c: (mdp.h - 1 - c[1]) * cell
    vals = [abs(v) for v in (V or {}).values()] + [abs(q) for d in (Q or {}).values() for q in d.values()]
    vmax = max(vals + [abs(v) for v in mdp.exits.values()] + [1e-9])
    fs = max(11, cell // 6)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-2 -2 {W + 4} {H + 4}" width="{W + 4}" '
         f'style="max-width:100%;height:auto;background:{BG};border-radius:10px;font-family:sans-serif">',
         '<defs>' + ''.join(f'<marker id="{uid}h{i}" markerWidth="5" markerHeight="5" refX="4" refY="2.5" markerUnits="strokeWidth" viewBox="0 0 5 5" preserveAspectRatio="none" '
                            f'orient="auto"><path d="M0,0 L5,2.5 L0,5 z" fill="{col}"/></marker>'
                            for i, col in enumerate((TXT, ORANGE, '#94a3b8'))) + '</defs>']
    t = lambda x, y, s, col=TXT, size=fs, w=600: o.append(
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{col}" font-size="{size}" font-weight="{w}" text-anchor="middle" '
        f'dominant-baseline="central">{s}</text>')
    for c in [(x, y) for x in range(mdp.w) for y in range(mdp.h)]:
        x, y = X(c), Y(c)
        if c in mdp.walls:
            o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{WALL}" stroke="{BG}" stroke-width="2"/>')
            continue
        ex = c in mdp.exits
        if Q is not None and not ex and c in Q:
            cx, cy = x + cell / 2, y + cell / 2
            corners = {'N': ((x, y), (x + cell, y)), 'S': ((x, y + cell), (x + cell, y + cell)),
                       'E': ((x + cell, y), (x + cell, y + cell)), 'W': ((x, y), (x, y + cell))}
            best = max(Q[c].values())
            for a, (p1, p2) in corners.items():
                q = Q[c][a]
                o.append(f'<polygon points="{p1[0]},{p1[1]} {p2[0]},{p2[1]} {cx},{cy}" fill="{_shade(q, vmax)}" '
                         f'stroke="{LINE}" stroke-width="1"/>')
                tx, ty = (p1[0] + p2[0] + cx) / 3, (p1[1] + p2[1] + cy) / 3
                t(tx, ty, fmt(q), ORANGE if abs(q - best) < 1e-9 else TXT, round(fs * 0.8))
            continue
        v = Q[c]['X'] if (Q is not None and ex and c in Q) else (V or {}).get(c)
        fill = _shade(mdp.exits[c] if ex and V is None else v, vmax) if ((ex and V is None) or v is not None) else '#0f172a'
        o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{LINE}" stroke-width="1.5"/>')
        if ex:
            o.append(f'<rect x="{x + 6}" y="{y + 6}" width="{cell - 12}" height="{cell - 12}" fill="none" '
                     f'stroke="{TXT}" stroke-width="1.5"/>')
        if v is not None or (ex and V is None):
            shown = v if v is not None else mdp.exits[c]
            t(x + cell / 2, y + cell / 2 + (cell * 0.18 if pol and not ex else 0),
              ('%+g' % shown) if (ex and V is None) else fmt(shown), size=round(fs * 1.1))
        if show_names and c in mdp.names:
            t(x + cell * 0.14, y + cell * 0.16, mdp.names[c], '#94a3b8', round(fs * 0.85))
    if pol:
        for c, a in pol.items():
            if a == 'X' or c in mdp.walls:
                if a == 'X' and mdp.walk_from_exits:
                    t(X(c) + cell / 2, Y(c) + cell * 0.82, 'exit', ORANGE, round(fs * 0.8))
                continue
            cx, cy = X(c) + cell / 2, Y(c) + cell / 2 - (cell * 0.1 if V is not None else 0)
            k = 0.22
            if c in mdp.exits:  # walking out of an exit cell (row layout): small arrow above the number
                cy, k = Y(c) + cell * 0.22, 0.16
            dx, dy = DIRS[a][0] * cell * k, -DIRS[a][1] * cell * k
            o.append(f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy:.1f}" '
                     f'stroke="{TXT}" stroke-width="{max(2, cell // 30)}" marker-end="url(#{uid}h0)"/>')
    for c in changed:  # cells that changed in the last step: orange frame
        o.append(f'<rect x="{X(c) + 3}" y="{Y(c) + 3}" width="{cell - 6}" height="{cell - 6}" fill="none" '
                 f'stroke="{ORANGE}" stroke-width="{max(3, cell // 25)}"/>')
    for c in read:  # cells a Bellman step reads: light-blue dashed frame
        o.append(f'<rect x="{X(c) + 8}" y="{Y(c) + 8}" width="{cell - 16}" height="{cell - 16}" fill="none" '
                 f'stroke="{READ}" stroke-width="{max(3, cell // 25)}" stroke-dasharray="7,4"/>')
    for i, c in enumerate(trail):
        o.append(f'<circle cx="{X(c) + cell / 2}" cy="{Y(c) + cell / 2}" r="{cell * 0.06}" fill="{AGENT}" '
                 f'opacity="{0.25 + 0.5 * (i + 1) / len(trail):.2f}"/>')
    arrow = lambda x1, y1, x2, y2, col, dash, mk: o.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
        f'stroke-width="{max(3, cell // 22)}" stroke-dasharray="{dash}" marker-end="url(#{uid}h{mk})"/>')
    if agent and prev:
        px, py = X(prev) + cell / 2, Y(prev) + cell / 2
        if intended and intended in DIRS:
            ix, iy = DIRS[intended]
            arrow(px + ix * cell * .2, py - iy * cell * .2, px + ix * cell * .47, py - iy * cell * .47, '#94a3b8', '5,4', 2)
        if actual and actual in DIRS:
            if agent != prev:
                ax, ay = X(agent) + cell / 2, Y(agent) + cell / 2
                arrow(px + DIRS[actual][0] * cell * .2, py - DIRS[actual][1] * cell * .2,
                      ax - DIRS[actual][0] * cell * .24, ay + DIRS[actual][1] * cell * .24, ORANGE, '', 1)
            else:  # bumped into a wall: short arrow toward it and back
                ex, ey = px + DIRS[actual][0] * cell * .46, py - DIRS[actual][1] * cell * .46
                arrow(px + DIRS[actual][0] * cell * .22, py - DIRS[actual][1] * cell * .22, ex, ey, ORANGE, '3,3', 1)
                t(ex - DIRS[actual][0] * cell * .12, ey + DIRS[actual][1] * cell * .12 + (cell * .14 if DIRS[actual][1] == 0 else 0),
                  'bump', ORANGE, round(fs * 0.75))
    if agent:
        cx, cy = X(agent) + cell / 2, Y(agent) + cell / 2
        o.append(f'<circle cx="{cx}" cy="{cy}" r="{cell * 0.2}" fill="{AGENT}" stroke="{BG}" stroke-width="2"/>')
    o.append('</svg>')
    return ''.join(o)


def html_player(mdp, frames, cell=90, pol=None, V=None, speed=3, uid='mp'):
    """Self-contained HTML player for one episode: ▶ / ⏸ / step. Grey dashed arrow = intended move,
    orange arrow = what actually happened."""
    pics = [grid_svg(mdp, cell, V=V, pol=pol, agent=frames[0]['s'], uid=uid + '0')]
    caps = ['t = 0 · start']
    for i, f in enumerate(frames):
        slip = f['d'] != f['a'] and f['a'] != 'X'
        nxt = f['n'] or f['s']
        pics.append(grid_svg(mdp, cell, V=V, pol=pol, agent=nxt, prev=f['s'], intended=f['a'] if slip else None,
                             actual=f['d'] if f['n'] else None, trail=[g['s'] for g in frames[max(0, i - 6):i]],
                             uid=f'{uid}{i + 1}'))
        what = (f"exit: +{f['r']:g}" if f['r'] >= 0 else f"exit: {f['r']:g}") if f['a'] == 'X' else \
            (f"chose {ARROW[f['a']]}, slipped {ARROW[f['d']]}" if slip else f"chose {ARROW[f['a']]}") + \
            (" (bumped into a wall)" if f['n'] == f['s'] else "")
        caps.append(f"t = {f['t'] + 1} · {what} · reward {f['r']:g} · return so far {f['ret']:.2f}")
    btn = ('style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:3px 12px;'
           'cursor:pointer;font-size:15px;font-weight:600"')
    return f"""<div style="background:{BG};color:{TXT};padding:10px;border-radius:12px;font-family:sans-serif">
<div style="display:flex;gap:6px;align-items:center;margin-bottom:8px">
<button id="{uid}p" {btn}>▶</button><button id="{uid}b" {btn}>◀</button><button id="{uid}n" {btn}>▶|</button>
<button id="{uid}r" {btn}>⟲</button><span id="{uid}c" style="font-size:15px;font-weight:600;margin-left:8px"></span></div>
<div id="{uid}v"></div></div>
<script>(function(){{
const P={json.dumps(pics)}, C={json.dumps(caps)}; let k=0, timer=null;
const $=s=>document.getElementById("{uid}"+s);
function show(){{ $("v").innerHTML=P[k]; $("c").textContent=C[k]; }}
function stop(){{ clearInterval(timer); timer=null; $("p").textContent="▶"; }}
$("p").onclick=()=>{{ if(timer) return stop(); if(k>=P.length-1) k=0; $("p").textContent="⏸";
  timer=setInterval(()=>{{ if(k>=P.length-1) return stop(); k++; show(); }}, {round(1000 / speed)}); }};
$("n").onclick=()=>{{ stop(); k=Math.min(P.length-1,k+1); show(); }};
$("b").onclick=()=>{{ stop(); k=Math.max(0,k-1); show(); }};
$("r").onclick=()=>{{ stop(); k=0; show(); }};
show(); }})();</script>"""


# ---------------- self-check ----------------
if __name__ == '__main__':
    m = GridMDP('book', gamma=0.9)
    V = value_iteration(m)
    expect = {(0, 2): .64, (1, 2): .74, (2, 2): .85, (0, 1): .57, (2, 1): .57,
              (0, 0): .49, (1, 0): .43, (2, 0): .48, (3, 0): .28}
    assert all(round(V[-1][c], 2) == v for c, v in expect.items()), V[-1]          # CS188 lec08 V* snapshot
    assert round(value_iteration(m, 2)[2][(2, 2)], 2) == 0.72                        # CS188 V_2 snapshot
    # lesson 6 deck policies (noise 0.2, no discount), as computed by g_policy in lesson6_content.py
    deck = {-0.01: ('→→→X', '↑X←X', '↑←←↓'), -0.03: ('→→→X', '↑X↑X', '↑←←←'),
            -0.4: ('→→→X', '↑X↑X', '↑→↑←'), -2.0: ('→→→X', '↑X→X', '→→→↑')}
    for r, rows in deck.items():
        mm = GridMDP('book', living=r, gamma=1.0)
        p = greedy_policy(mm, value_iteration(mm)[-1])
        got = tuple(''.join('X' if (x, y) in mm.walls else ('X' if p[(x, y)] == 'X' else ARROW[p[(x, y)]])
                            for x in range(4)) for y in (2, 1, 0))
        assert got == rows, (r, got)
    # discount quiz: gamma 1 -> everyone West (e too); gamma 0.1 -> a, e exit, b c West, d East; tie at 1/sqrt(10)
    q = lambda g: ''.join(greedy_policy(GridMDP('row', gamma=g), value_iteration(GridMDP('row', gamma=g))[-1])[(x, 0)]
                          for x in range(5))
    assert q(1.0) == 'XWWWW' and q(0.1) == 'XWWEX', (q(1.0), q(0.1))
    assert q(0.30)[3] == 'E' and q(0.33)[3] == 'W'
    # lesson 7 worked example (corridor): the hand-computed table, converged at k = 17
    cv = value_iteration(GridMDP('corridor'))
    assert [[round(V[(x, 0)], 2) for x in (1, 2)] for V in cv[:5]] == [[0, 0], [0, 0], [0, .72], [.52, .85], [.71, .87]]
    assert len(cv) - 1 == 17 and (round(cv[-1][(1, 0)], 3), round(cv[-1][(2, 0)], 3)) == (.771, .878)
    # explain() adds up to q(), and V_k does not depend on the order the cells are visited in
    for k in (1, 2, 3):
        Vp = value_iteration(m, k)
        for order in (m.cells, m.cells[::-1], sorted(m.cells, key=lambda c: (c[0] * 7 + c[1] * 3) % 5)):
            Vn = {}
            for c in order:
                ex = m.explain(Vp[-2], c)
                for a, terms, qv in ex:
                    assert abs(sum(p * (r + (m.gamma * v if n else 0.0)) for p, n, r, v in terms) - qv) < 1e-12
                Vn[c] = max(qv for _, _, qv in ex)
            assert Vn == Vp[-1], order
    # policy iteration reaches the value-iteration policy
    for name in LAYOUTS:
        mm = GridMDP(name)
        Vv = value_iteration(mm)[-1]
        pi_pol, pi_V = policy_iteration(mm)[-1]
        assert all(abs(pi_V[s] - Vv[s]) < 1e-6 for s in mm.cells), name
    # episodes are reproducible and noise really slips
    fr, end = run_episode(m, greedy_policy(m, V[-1]), seed=3)
    assert (fr, end) == run_episode(m, greedy_policy(m, V[-1]), seed=3) and end in (1.0, -1.0)
    st = run_many(m, greedy_policy(m, V[-1]), n=200)
    assert sum(st['ends'].values()) == 200
    print('ok', st)
