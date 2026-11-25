"""The expectimax tree behind V_k, and the same computation folded into layers (Value iteration).
Used by the "Tree" tab of app.py. Self-check: python3 mdp_tree.py"""
from functools import lru_cache
from mdp_engine import ARROW, AGENT, BG, LINE, ORANGE, TXT, WALL, GridMDP, fmt, grid_svg, value_iteration

BLUE, GREY, MUTED = '#3b82f6', '#64748b', '#94a3b8'


def acts_fn(m, pol=None):
    """Actions at a state node: all of them (max = Value iteration) or only pol[s] (policy evaluation)."""
    return (lambda s: [pol[s]]) if pol else m.actions


def outs(m, s, a):
    """[(p, next state or None)], slips into the same cell (a wall bump) merged into one branch."""
    d = {}
    for p, _, n in m.outcomes(s, a):
        d[n] = d.get(n, 0) + p
    return [(p, n) for n, p in d.items()]


def build(m, s, k, acts):
    """Expectimax tree with k steps left. S = state (max), Q = chance node, T = episode over."""
    node = dict(kind='S', s=s, k=k, ch=[])
    if k:
        for a in acts(s):
            node['ch'].append(dict(kind='Q', s=s, a=a, k=k, r=m.reward(s, a),
                                   ch=[(p, build(m, n, k - 1, acts) if n else dict(kind='T', k=k - 1, ch=[]))
                                       for p, n in outs(m, s, a)]))
    return node


def evaluate(m, node):
    """Fills node['v'] bottom-up (and 'best' on state nodes); returns the root value."""
    if node['kind'] == 'T':
        node['v'] = 0.0
    elif node['kind'] == 'Q':
        node['v'] = sum(p * (node['r'] + m.gamma * evaluate(m, c)) for p, c in node['ch'])
    else:
        vals = [evaluate(m, q) for q in node['ch']]
        node['v'] = max(vals) if vals else 0.0
        node['best'] = vals.index(node['v']) if vals else None
    return node['v']


def shown(node, step):
    """Build steps 0..2k+1: 1 = the leaves (V_0 = 0), then per level: chance nodes (average), state nodes (max)."""
    if node['kind'] == 'T':
        return step >= 1
    return step >= (2 * node['k'] if node['kind'] == 'Q' else 1 + 2 * node['k'])


def tree_size(m, s, k, acts):
    @lru_cache(None)
    def n(s, k):
        return 1 + (sum(n(x, k - 1) for a in acts(s) for _, x in outs(m, s, a) if x) if k else 0)
    return n(s, k)


def layers(m, k, acts):
    """V_0 .. V_k with the given actions (all = Value iteration)."""
    Vs = [{s: 0.0 for s in m.cells}]
    for _ in range(k):
        V = Vs[-1]
        Vs.append({s: max(sum(p * (m.reward(s, a) + (m.gamma * V[n] if n else 0)) for p, n in outs(m, s, a))
                          for a in acts(s)) for s in m.cells})
    return Vs


# ---------------- drawing ----------------
def mini_board(m, s, x, y, c, frame, width=2.5):
    """Tiny board with the agent in cell s (as the boards in the lesson 5 search tree)."""
    o = []
    for gx in range(m.w):
        for gy in range(m.h):
            px, py = x + gx * c, y + (m.h - 1 - gy) * c
            if (gx, gy) in m.walls:
                fill = WALL
            elif (gx, gy) in m.exits:
                fill = '#166534' if m.exits[(gx, gy)] > 0 else '#991b1b'
            else:
                fill = '#0f172a'
            o.append(f'<rect x="{px}" y="{py}" width="{c}" height="{c}" fill="{fill}" stroke="{LINE}" stroke-width="0.6"/>')
    o.append(f'<circle cx="{x + (s[0] + .5) * c}" cy="{y + (m.h - .5 - s[1]) * c}" r="{c * .34}" fill="{AGENT}"/>')
    o.append(f'<rect x="{x - 2}" y="{y - 2}" width="{m.w * c + 4}" height="{m.h * c + 4}" fill="none" '
             f'stroke="{frame}" stroke-width="{width}" rx="3"/>')
    return ''.join(o)


def tree_svg(m, root, step, hl=None, font=15):
    """Top-down tree: state nodes = mini boards (blue frame = max), chance nodes = circles with the action.
    Values appear as the build step reaches their level; the winning branch turns orange."""
    c = 11 if m.w * m.h <= 16 else 7
    bw, bh = m.w * c, m.h * c
    gap, rq = 16, 15
    S_ROW = bh + font + 30               # board + value line
    Q_ROW = 2 * rq + font + 24           # circle + probability line
    leaves = []

    def place(n, depth_y):
        n['y'] = depth_y
        if n['kind'] == 'S' and n['ch']:
            for q in n['ch']:
                q['y'] = depth_y + S_ROW
                for _, ch in q['ch']:
                    place(ch, depth_y + S_ROW + Q_ROW)
                q['x'] = sum(ch['x'] for _, ch in q['ch']) / len(q['ch'])
            n['x'] = sum(q['x'] for q in n['ch']) / len(n['ch'])
        else:
            n['x'] = len(leaves) * (bw + gap) + bw / 2
            leaves.append(n)

    place(root, 0)
    W = len(leaves) * (bw + gap) + 90
    H = max(n['y'] for n in leaves) + S_ROW + 10
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-70 -10 {W} {H}" data-w="{W}" '
         f'style="background:{BG};font-family:sans-serif">']
    t = lambda x, y, s, col=TXT, size=font, w=700, anchor='middle': o.append(
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{col}" font-size="{size}" font-weight="{w}" '
        f'text-anchor="{anchor}" dominant-baseline="central" stroke="{BG}" stroke-width="4" '
        f'paint-order="stroke">{s}</text>')
    nodes = []

    def walk(n):
        nodes.append(n)
        for ch in n['ch']:
            walk(ch[1] if isinstance(ch, tuple) else ch)
    walk(root)
    for j in range(root['k'] + 1):  # row labels on the left: V_k, Q, V_k-1, ...
        y = (root['k'] - j) * (S_ROW + Q_ROW)
        t(-40, y + bh / 2, f'V{sub(j)}', MUTED, font, 700)
        if j:
            t(-40, y + S_ROW, f'Q{sub(j)}', MUTED, font, 700)
    # edges first
    for n in nodes:
        if n['kind'] == 'S':
            for i, q in enumerate(n['ch']):
                win = shown(n, step) and i == n.get('best')
                o.append(f'<line x1="{n["x"]}" y1="{n["y"] + bh + font * 1.6}" x2="{q["x"]}" y2="{q["y"] - rq}" '
                         f'stroke="{ORANGE if win else GREY}" stroke-width="{3 if win else 1.5}"/>')
        elif n['kind'] == 'Q':
            for p, ch in n['ch']:
                o.append(f'<line x1="{n["x"]}" y1="{n["y"] + rq}" x2="{ch["x"]}" y2="{ch["y"] - font * 1.1}" '
                         f'stroke="{GREY}" stroke-width="1.2" stroke-dasharray="4,3"/>')
                lbl = f'{p:g}' if ch['kind'] == 'S' else f'R = {n["r"]:+g}'
                t(ch['x'], ch['y'] - font * .55, lbl, MUTED, round(font * .8), 600)
    for n in nodes:
        x, y = n['x'], n['y']
        if n['kind'] == 'S':
            frame = ORANGE if hl is not None and n['s'] == hl else BLUE
            o.append(mini_board(m, n['s'], x - bw / 2, y, c, frame, 4 if frame == ORANGE else 2.5))
            t(x, y + bh + font * 0.9, fmt(n['v']) if shown(n, step) else '?', TXT if shown(n, step) else MUTED)
        elif n['kind'] == 'Q':
            o.append(f'<circle cx="{x}" cy="{y}" r="{rq}" fill="#1e293b" stroke="{MUTED}" stroke-width="1.5"/>')
            t(x, y, 'X' if n['a'] == 'X' else ARROW[n['a']], TXT, round(font * 1.05))
            if shown(n, step):
                t(x, y - rq - font * .55, fmt(n['v']), '#fde68a', round(font * .85), 700)
        else:
            t(x, y + font * .6, 'end', MUTED, round(font * .85))
    o.append('</svg>')
    return ''.join(o)


def sub(k):
    return ''.join('₀₁₂₃₄₅₆₇₈₉'[int(d)] for d in str(k))


def layers_svg(m, Vs, step, root, acts, cell=60):
    """V_0 at the bottom up to V_k on top. Layers above `step` are still empty. On the layer being computed,
    the root cell (orange) reads only its next cells one layer down (framed; orange = the best action's outcomes,
    with their probabilities), and its Q-values are listed beside it."""
    k = len(Vs) - 1
    W, H, gap, left, right = m.w * cell, m.h * cell, 50, 70, 230
    top = lambda j: (k - j) * (H + gap)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -10 {left + W + right} {(k + 1) * (H + gap)}" '
         f'data-w="{left + W + right}" style="background:{BG};font-family:sans-serif">']
    t = lambda x, y, s, col=TXT, size=18, anchor='middle': o.append(
        f'<text x="{x}" y="{y}" fill="{col}" font-size="{size}" font-weight="700" text-anchor="{anchor}" '
        f'dominant-baseline="central">{s}</text>')
    for j in range(k + 1):
        g = grid_svg(m, cell, V=Vs[j] if j <= step else None, uid=f'L{j}',
                     changed=[root] if j == step and j else ())
        o.append(g.replace('<svg ', f'<svg x="{left}" y="{top(j)}" height="{H + 4}" ', 1))
        t(left / 2, top(j) + H / 2, f'V{sub(j)}', TXT, 22)
    if step:
        V = Vs[step - 1]
        q = {a: sum(p * (m.reward(root, a) + (m.gamma * V[n] if n else 0)) for p, n in outs(m, root, a))
             for a in acts(root)}
        best = max(q, key=q.get)
        X = lambda s: left + 2 + s[0] * cell
        Y = lambda s: top(step - 1) + 2 + (m.h - 1 - s[1]) * cell
        read = {n for a in q for _, n in outs(m, root, a) if n}
        for n in read - {n for _, n in outs(m, root, best)}:
            o.append(f'<rect x="{X(n) + 4}" y="{Y(n) + 4}" width="{cell - 8}" height="{cell - 8}" fill="none" '
                     f'stroke="{MUTED}" stroke-width="2.5" stroke-dasharray="6,4"/>')
        for p, n in outs(m, root, best):
            if n:
                o.append(f'<rect x="{X(n) + 4}" y="{Y(n) + 4}" width="{cell - 8}" height="{cell - 8}" fill="none" '
                         f'stroke="{ORANGE}" stroke-width="4"/>')
                t(X(n) + cell * .26, Y(n) + cell * .2, f'{p:g}', ORANGE, 13)
        x0, y0 = left + W + 24, top(step) + 14
        t(x0, y0, f'V{sub(step)}{root}:', TXT, 16, 'start')
        for i, (a, v) in enumerate(q.items()):
            t(x0, y0 + 24 * (i + 1), f'{"exit" if a == "X" else ARROW[a]}  {fmt(v)}', ORANGE if a == best else MUTED,
              16, 'start')
        t(x0, y0 + 24 * (len(q) + 1), ('max = ' if len(q) > 1 else '= ') + fmt(q[best]), ORANGE, 16, 'start')
    o.append('</svg>')
    return ''.join(o)


def zoom_html(svg, height=620, font=16):
    """Scrollable panel with − / + / fit / 1:1 (redraws the SVG at the new width, so text stays sharp)."""
    return f"""<div style="font-family:sans-serif">
<div style="margin-bottom:6px">{''.join(f'<button onclick="z({a})" style="font-size:{font}px;margin-right:4px">{b}</button>'
                                       for a, b in (("0.8", "−"), ("1.25", "+"), ("'fit'", "fit"), ("'one'", "1:1")))}</div>
<div id="p" style="overflow:auto;height:{height}px;border:1px solid {LINE};border-radius:8px;background:{BG}">{svg}</div></div>
<script>
const p = document.getElementById('p'), s = p.querySelector('svg'), w0 = +s.dataset.w;
const pw = () => (p.clientWidth || 1e9) - 4;  // 0 while the tab is hidden
let w = w0; s.style.width = w + 'px';  // start at 1:1, scrolled to the root (the middle)
requestAnimationFrame(() => {{ p.scrollLeft = (w0 - p.clientWidth) / 2; }});
function z(a) {{ w = a === 'fit' ? Math.min(pw(), 4 * w0) : a === 'one' ? w0 : w * a; s.style.width = w + 'px'; }}
</script>"""


if __name__ == '__main__':
    for name in ('book', 'bridge', 'discount'):
        m = GridMDP(name)
        Vs = value_iteration(m, k=3)
        for s in m.cells:
            for k in range(4 if name == 'book' else 3):
                assert abs(evaluate(m, build(m, s, k, m.actions)) - Vs[k][s]) < 1e-9, (name, s, k)
        assert all(abs(a[s] - b[s]) < 1e-9 for a, b in zip(layers(m, 3, m.actions), Vs) for s in m.cells)
    m = GridMDP('book')
    assert abs(evaluate(m, build(m, (2, 2), 2, m.actions)) - 0.72) < 1e-9
    assert [tree_size(m, (2, 2), k, m.actions) for k in (1, 2, 3)] == [13, 115, 1063]
    assert [tree_size(m, (0, 0), k, m.actions) for k in (1, 2, 3)] == [11, 111, 1129]
    print('mdp_tree ok')
