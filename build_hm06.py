"""Build HM_06.ipynb (student) and HM_06_solution.ipynb from one cell list.
Usage: python build_hm06.py <engine mdp_engine.py> <out_dir> [answers.json]"""
import ast, json, random, sys
import nbformat as nbf

ENGINE_PATH, OUT = sys.argv[1], sys.argv[2]
ANSWERS = json.load(open(sys.argv[3], encoding='utf-8')) if len(sys.argv) > 3 and sys.argv[3] != '--numbers' else {}
sys.path.insert(0, ENGINE_PATH.rsplit('/', 1)[0])
import mdp_engine as me

# ---------- engine for the notebook: no transition model, no sampling, no planning (that is the exercise) ----------
src = open(ENGINE_PATH, encoding='utf-8').read()
tree = ast.parse(src)
DROP = {'value_iteration', 'q_values', 'greedy_policy', 'policy_evaluation', 'policy_iteration', 'sample',
        'run_episode', 'run_many'}
parts = []
for node in tree.body:
    if isinstance(node, (ast.If, ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and node is tree.body[0]):
        continue
    if getattr(node, 'name', None) in DROP:
        continue
    seg = ast.get_source_segment(src, node)
    if isinstance(node, ast.ClassDef):
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name in ('outcomes', 'q'):
                seg = seg.replace('    ' + ast.get_source_segment(src, m), '')
    parts.append(seg)
engine = '\n\n'.join(parts)
assert 'def outcomes' not in engine and 'def q(' not in engine and 'value_iteration' not in engine

# ---------- expected values (computed with the reference engine) ----------
def opt(m):
    return me.greedy_policy(m, me.value_iteration(m)[-1])

SETTINGS = ([('book', 0.2, 0.0, 0.9), ('book', 0.0, 0.0, 0.9)] +
            [('book', 0.2, r, 1.0) for r in (0.0, -0.01, -0.03, -0.4, -2.0)] +
            [('row', 0.0, 0.0, g) for g in (1.0, 0.5, 0.33, 0.3, 0.2, 0.1)] +
            [('bridge', n, 0.0, 0.9) for n in (0.0, 0.2)])
key = lambda m: (m.name, round(m.noise, 3), round(m.living, 3) + 0.0, round(m.gamma, 3))
OPTIMAL = {}
for name, n, r, g in SETTINGS:
    m = me.GridMDP(name, n, r, g)
    OPTIMAL[key(m)] = opt(m)

def merged(outs):
    d = {}
    for p, _, nxt in outs:
        d[nxt] = round(d.get(nxt, 0) + p, 9)
    return d
EXPECT_OUT = {}
for name in ('book', 'bridge', 'row'):
    for noise in (0.0, 0.2, 0.4):
        m = me.GridMDP(name, noise)
        EXPECT_OUT[(name, noise)] = {(s, a): merged(m.outcomes(s, a)) for s in m.cells for a in m.actions(s)}

def ref_ev(t, kind='max'):
    if not isinstance(t, list):
        return t
    if kind == 'max':
        return max(ref_ev(c, 'chance') for c in t)
    kids = [c if isinstance(c, tuple) else (1 / len(t), c) for c in t]
    return sum(p * ref_ev(c, 'max') for p, c in kids)

def rand_tree(rng, b, d, kind='max'):
    if d == 0:
        return rng.randint(0, 99)
    kids = [rand_tree(rng, b, d - 1, 'chance' if kind == 'max' else 'max') for _ in range(b)]
    if kind == 'chance' and rng.random() < 0.5:
        w = [rng.randint(1, 5) for _ in kids]
        kids = [(x / sum(w), k) for x, k in zip(w, kids)]
    return kids
R = random.Random(6)
RANDOM_TREES = [rand_tree(R, b, d) for b, d in [(2, 2), (3, 2), (2, 4), (3, 3), (4, 2)]]
EXPECT_RANDOM = [ref_ev(t) for t in RANDOM_TREES]

HAND = {(0, 0): 'E', (1, 0): 'E', (2, 0): 'N', (3, 0): 'W', (0, 1): 'S', (2, 1): 'N', (0, 2): 'E', (1, 2): 'E',
        (2, 2): 'E', (3, 2): 'X', (3, 1): 'X'}
REF_EVAL = {}
for r in (-0.01, -0.03, -0.4, -2.0):
    m = me.GridMDP('book', 0.2, r, 1.0)
    for label, pol in (('optimal', opt(m)), ('hand', HAND)):
        x = me.run_many(m, pol, 200)
        REF_EVAL[(r, label)] = (round(x['avg_return'], 2), {k: v for k, v in x['ends'].items()}, round(x['avg_steps'], 1))
m = me.GridMDP('book')
REF_BOOK = round(me.run_many(m, opt(m), 100)['avg_return'], 3)
if __name__ == '__main__' and '--numbers' in sys.argv:
    print('REF_BOOK', REF_BOOK)
    for k, v in REF_EVAL.items():
        print(k, v)
    sys.exit()

SETUP = r'''#@title 🔧 תא הגדרות: הריצו אותו ראשון (אין צורך לקרוא את הקוד)
import json, math, random, signal
from fractions import Fraction
from statistics import mean
from IPython.display import HTML, display
import matplotlib.pyplot as plt

# ============================================================
# מנוע "עולם הרשת" (אותו מנוע כמו בסימולטור של הקורס, demos/mdp)
# ============================================================
''' + engine + r'''


def show_grid(mdp, policy=None, values=None, cell=80):
    """ציור הלוח. policy: מילון {משבצת: פעולה} (חיצים). values: מילון {משבצת: מספר}."""
    display(HTML(grid_svg(mdp, cell, V=values, pol=policy)))

def animate(mdp, frames, policy=None, cell=80):
    """אנימציה של ריצה: ▶ מתחיל. חץ אפור מקווקו = מה שהסוכן בחר, חץ כתום = מה שקרה בפועל."""
    display(HTML(html_player(mdp, frames, cell, pol=policy)))


# ============================================================
# מדיניות אופטימלית מוכנה ("קופסה שחורה")
# בשיעור 7 נלמד לחשב אותה בעצמנו (Value Iteration). כאן היא מחושבת מראש, רק להגדרות שבתרגיל.
# ============================================================
OPTIMAL = ''' + repr(OPTIMAL) + r'''

def optimal_policy(mdp):
    key = (mdp.name, round(mdp.noise, 3), round(mdp.living, 3) + 0.0, round(mdp.gamma, 3))
    if key not in OPTIMAL:
        raise KeyError(f'אין מדיניות מוכנה עבור {key}. בתרגיל יש מדיניות אופטימלית רק להגדרות שבשאלות. '
                       'בשיעור 7 נלמד לחשב אותה בעצמנו.')
    return dict(OPTIMAL[key])

HAND = ''' + repr(HAND) + r'''   # "הדרך הקצרה": לאורך השורה התחתונה, ואז למעלה ליד הבור


# ============================================================
# עצי Expectimax מופשטים (כמו בהרצאה)
# ============================================================
def is_leaf(t):
    return not isinstance(t, list)

def chance_children(t):
    """הילדים של צומת מקריות, עם ההסתברויות: [(p, child), ...]. בלי הסתברויות נתונות: התפלגות אחידה."""
    return [c if isinstance(c, tuple) else (1 / len(t), c) for c in t]

def minimax(t, kind='max'):
    """Minimax על אותם עצים: צמתי המקריות מתפרשים כ-MIN (ההסתברויות לא משנות)."""
    if is_leaf(t):
        return t
    kids = [c[1] if isinstance(c, tuple) else c for c in t]
    return max(minimax(c, 'min') for c in kids) if kind == 'max' else min(minimax(c, 'max') for c in kids)

def best_child(t, value_fn):
    """אינדקס הילד הטוב ביותר של שורש MAX, לפי value_fn (ילד של MAX הוא צומת מקריות / MIN)."""
    vals = [value_fn(c, 'chance') if value_fn is not minimax else minimax(c, 'min') for c in t]
    return max(range(len(t)), key=lambda i: vals[i]), vals

def square_leaves(t):
    if is_leaf(t):
        return t * t
    return [(c[0], square_leaves(c[1])) if isinstance(c, tuple) else square_leaves(c) for c in t]

LECTURE_TREES = {
    'full example (3 12 9 | 2 4 6 | 15 6 0)': ([[3, 12, 9], [2, 4, 6], [15, 6, 0]], 'max', 8),
    'one chance node (8 24 -12, p = 1/2 1/3 1/6)': ([(1 / 2, 8), (1 / 3, 24), (1 / 6, -12)], 'chance', 10),
    'lec07 (10 10 | 9 100)': ([[10, 10], [9, 100]], 'max', 54.5),
    'magnitudes (0 40 | 20 30)': ([[0, 40], [20, 30]], 'max', 25),
}


def _kids(t):
    return [c[1] if isinstance(c, tuple) else c for c in t]

def _tlayout(t, path=(), xy=None, counter=None):
    xy = {} if xy is None else xy; counter = counter or [0]
    if is_leaf(t):
        xy[path] = (counter[0], -len(path)); counter[0] += 1
    else:
        xs = [_tlayout(c, path + (i,), xy, counter)[path + (i,)][0] for i, c in enumerate(_kids(t))]
        xy[path] = ((xs[0] + xs[-1]) / 2, -len(path))
    return xy

def _sub(t, path):
    for i in path:
        t = _kids(t)[i]
    return t

def show_tree(t, kind='max', value_fn=None, title='', as_min=False, ax=None):
    """מצייר עץ Expectimax. ▲ צומת MAX, ○ צומת מקריות (או ▽ צומת MIN כש-as_min=True), ■ עלה.
    על הקשתות מתחת לצומת מקריות: ההסתברויות. value_fn (למשל expectimax או minimax): הערך נכתב בכל צומת,
    והקשת שבוחר השורש מודגשת בירוק."""
    xy = _tlayout(t)
    kinds = {}
    def walk(n, path, k):
        kinds[path] = k
        if not is_leaf(n):
            for i, c in enumerate(_kids(n)):
                walk(c, path + (i,), 'chance' if k == 'max' else 'max')
    walk(t, (), kind)
    vals = {}
    if value_fn is not None:
        for path, k in kinds.items():
            n = _sub(t, path)
            vals[path] = n if is_leaf(n) else value_fn(n, 'min' if (as_min and k == 'chance') else k)
    best = None
    if vals and kind == 'max' and not is_leaf(t):
        kids = range(len(t))
        best = max(kids, key=lambda i: vals[(i,)])
    width = max(x for x, _ in xy.values()) + 1
    depth = max(len(p) for p in xy)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * width), 1.4 * depth + 1.6))
    for path, (x, y) in xy.items():
        if not path:
            continue
        px, py = xy[path[:-1]]
        green = best is not None and path == (best,)
        ax.plot([px, x], [py, y], color='#16a34a' if green else '#64748b', lw=4 if green else 1.5, zorder=1)
        parent = _sub(t, path[:-1])
        if kinds[path[:-1]] == 'chance' and not (as_min and value_fn is not None):
            p = chance_children(parent)[path[-1]][0]
            fr = Fraction(p).limit_denominator(12)
            side = 0 if abs(x - px) < 1e-9 else (1 if x > px else -1)
            lx, ly = px + (x - px) * 0.55, py + (y - py) * 0.55
            ax.text(lx + (0.12 if side >= 0 else -0.12), ly, f'{fr.numerator}/{fr.denominator}', fontsize=11,
                    color='#b45309', weight='bold', ha='left' if side >= 0 else 'right', va='center')
    fmt = lambda v: f'{v:g}' if isinstance(v, (int, float)) else str(v)
    for path, (x, y) in xy.items():
        n, k = _sub(t, path), kinds[path]
        if is_leaf(n):
            ax.scatter([x], [y], marker='s', s=1100, color='white', edgecolors='#334155', zorder=2)
            ax.text(x, y, fmt(n), ha='center', va='center', fontsize=14 if len(fmt(n)) <= 3 else 11, weight='bold',
                    zorder=3)
            continue
        if k == 'max':
            ax.scatter([x], [y], marker='^', s=1500, color='#fde68a', edgecolors='#334155', zorder=2)
        elif as_min:
            ax.scatter([x], [y], marker='v', s=1500, color='white', edgecolors='#334155', zorder=2)
        else:
            ax.scatter([x], [y], marker='o', s=1300, color='#eef1f6', edgecolors='#334155', zorder=2)
        if path in vals:
            txt = fmt(round(vals[path], 3))
            ax.text(x, y - (0.0 if k != 'max' else 0.06), txt, ha='center', va='center',
                    fontsize=12 if len(txt) <= 3 else 10, weight='bold', zorder=3)
    handles = [plt.Line2D([], [], marker='^', ls='', ms=12, mfc='#fde68a', mec='#334155', label='MAX'),
               plt.Line2D([], [], marker='v' if as_min else 'o', ls='', ms=12,
                          mfc='white' if as_min else '#eef1f6', mec='#334155', label='MIN' if as_min else 'chance'),
               plt.Line2D([], [], marker='s', ls='', ms=11, mfc='white', mec='#334155', label='leaf (utility)')]
    if kind == 'chance':
        handles = handles[1:]
    ax.legend(handles=handles, loc='upper center', fontsize=10, frameon=False, ncol=len(handles),
              bbox_to_anchor=(0.5, 0.02))
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlim(-0.7, width - 0.3); ax.set_ylim(-depth - 0.9, 0.5)
    if own:
        plt.show()

def compare_trees(t, title=''):
    """Minimax (צמתי מקריות כ-MIN) מול Expectimax על אותו עץ, זה לצד זה."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    show_tree(t, 'max', minimax, 'Minimax', as_min=True, ax=axes[0])
    show_tree(t, 'max', expectimax, 'Expectimax', ax=axes[1])
    fig.suptitle(title, fontsize=14, weight='bold'); plt.tight_layout(); plt.show()


# ============================================================
# תשתית בדיקות
# ============================================================
RESULTS = {}
RANDOM_TREES = ''' + repr(RANDOM_TREES) + r'''
EXPECT_RANDOM = ''' + repr(EXPECT_RANDOM) + r'''
EXPECT_OUT = ''' + repr(EXPECT_OUT) + r'''
REF_BOOK = ''' + repr(REF_BOOK) + r'''

class _Timeout(Exception):
    pass

def _call(fn, args, seconds=20):
    def _handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return 'ok', fn(*args)
    except _Timeout:
        return 'timeout', None
    except Exception as e:
        return 'error', e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def _report_problem(status, value):
    if status == 'timeout':
        print('❌ הפונקציה לא הסתיימה בזמן: כנראה רקורסיה בלי תנאי עצירה, או לולאה שלא נגמרת.')
    else:
        print(f'❌ הפונקציה זרקה שגיאה: {type(value).__name__}: {value}')
        print('   אם עדיין לא השלמתם את ה-TODO, זה צפוי. אחרת, קראו את הודעת השגיאה ובדקו את השורה הרלוונטית.')

def _finish(name, ok):
    RESULTS[name] = ok
    print(f'✅ {name}: כל הבדיקות עברו!' if ok else f'⛔ {name}: יש בדיקות שנכשלו. תקנו והריצו שוב.')

def _close(a, b, tol=1e-9):
    return isinstance(a, (int, float)) and abs(a - b) <= tol

def check_expectimax(fn):
    ok = True
    for label, (t, kind, exp) in LECTURE_TREES.items():
        status, got = _call(fn, (t, kind))
        if status != 'ok':
            _report_problem(status, got); ok = False; break
        if not _close(got, exp):
            print(f'❌ [{label}] קיבלנו {got!r}, אבל ערך ה-Expectimax הוא {exp}.')
            if kind == 'chance':
                print('   רמז: צומת מקריות מחזיר סכום של p · value(child). האם השתמשתם ב-chance_children?')
            ok = False
    if ok:
        status, got = _call(fn, (7, 'max'))
        if got != 7:
            print('❌ על עלה בודד (7) צריך להחזיר 7. האם בדקתם is_leaf קודם?'); ok = False
    if ok:
        for t, exp in zip(RANDOM_TREES, EXPECT_RANDOM):
            status, got = _call(fn, (t, 'max'))
            if status != 'ok' or not _close(got, exp, 1e-6):
                print(f'❌ על עץ אקראי קיבלנו {got!r} במקום {exp}. בדקו שהרמות מתחלפות: max ← chance ← max ...')
                ok = False; break
    _finish('expectimax', ok)

def _merged(outs):
    d = {}
    for p, _, nxt in outs:
        d[nxt] = round(d.get(nxt, 0) + p, 9)
    return d

def check_outcomes(fn):
    ok = True
    for (name, noise), table in EXPECT_OUT.items():
        m = GridMDP(name, noise)
        for (s, a), exp in table.items():
            status, got = _call(fn, (m, s, a))
            if status != 'ok':
                _report_problem(status, got); _finish('outcomes', False); return
            try:
                g = _merged(got)
            except Exception:
                print(f'❌ outcomes צריכה להחזיר רשימה של שלשות (p, d, next). קיבלנו {got!r}.'); _finish('outcomes', False); return
            if abs(sum(g.values()) - 1) > 1e-9:
                print(f'❌ [{name}, noise {noise}] ב-{s} עם {a}: סכום ההסתברויות {sum(g.values()):.3f}, ולא 1.'); ok = False
            elif any(abs(g.get(k, 0) - v) > 1e-9 for k, v in exp.items()):
                print(f'❌ [{name}, noise {noise}] ב-{s} עם {a}: קיבלנו {g}, אבל צריך {exp}.')
                if a == 'X':
                    print("   רמז: הפעולה 'X' (יציאה) מסיימת את הריצה: [(1.0, 'X', None)].")
                elif s in m.exits:
                    print('   רמז: בלוח row מותר ללכת גם ממשבצת יציאה.')
                else:
                    print('   רמז: 1 − noise לכיוון שנבחר, noise / 2 לכל צד (SIDE[a]). קיר או קצה: נשארים במקום (step_to).')
                ok = False
            elif any(n is not None and m.step_to(s, d) != n for _, d, n in got):
                print(f'❌ [{name}] ב-{s} עם {a}: d צריך להיות הכיוון שבו זזנו בפועל, כך ש-step_to(s, d) == next.'); ok = False
            if not ok:
                break
        if not ok:
            break
    _finish('outcomes', ok)

def check_episodes(ret_fn, sample_fn, run_fn):
    ok = True
    for rewards, g, exp in [([1, 2, 3], 0.5, 2.75), ([3, 2, 1], 0.5, 4.25), ([], 0.9, 0.0), ([1, 1, 1, 1], 1.0, 4.0),
                            ([0, 0, 0, 1], 0.9, 0.729)]:
        status, got = _call(ret_fn, (rewards, g))
        if status != 'ok':
            _report_problem(status, got); ok = False; break
        if not _close(got, exp):
            print(f'❌ discounted_return({rewards}, {g}) צריכה להחזיר {exp}, קיבלנו {got!r}.')
            print('   רמז: התגמול במקום t מוכפל ב-γ בחזקת t, והראשון (t = 0) לא מהוון.'); ok = False
    if ok:
        m = GridMDP('book', 0.2)
        rng = random.Random(0)
        status, picks = _call(lambda: [sample_fn(m, (0, 1), 'N', rng) for _ in range(4000)], ())
        if status != 'ok':
            _report_problem(status, picks); ok = False
        else:
            slip = sum(d != 'N' for d, _ in picks) / len(picks)
            print(f'"צפונה" מ-(0,1), ‏4000 דגימות: החלקה ב-{slip:.1%} מהמקרים (צריך בערך 20%)')
            if not 0.17 <= slip <= 0.23:
                print('❌ שכיחות ההחלקה רחוקה מ-20%. בדקו את צבירת ההסתברויות ב-sample.'); ok = False
    if ok:
        pol = optimal_policy(GridMDP('book', 0.0))
        status, res = _call(run_fn, (GridMDP('book', 0.0), pol, 0))
        if status != 'ok':
            _report_problem(status, res); ok = False
        else:
            frames, end = res
            if end != 1.0 or len(frames) != 6 or not _close(frames[-1]['ret'], 0.9 ** 5):
                print(f'❌ בלי רעש: צריך 5 צעדים + יציאה (6 צעדים ברשימה), סיום ב-+1 ותשואה 0.59049. '
                      f'קיבלנו {len(frames)} צעדים, סיום {end}, תשואה {frames[-1]["ret"] if frames else None}.'); ok = False
    if ok:
        m = GridMDP('book', 0.2)
        pol = optimal_policy(m)
        a, b = run_fn(m, pol, 7), run_fn(m, pol, 7)
        if a != b:
            print('❌ אותו seed צריך לתת את אותה ריצה. האם יצרתם random.Random(seed) והשתמשתם רק בו?'); ok = False
        ends = [run_fn(m, pol, i)[1] for i in range(50)]
        if any(e not in (1.0, -1.0) for e in ends):
            print('❌ עם המדיניות האופטימלית כל הריצות צריכות להסתיים ביציאה (+1 או −1).'); ok = False
    _finish('episodes', ok)

def check_evaluate(fn):
    if not ready('episodes'):
        _finish('evaluate', False); return
    m = GridMDP('book')
    status, res = _call(fn, (m, optimal_policy(m), 100), seconds=60)
    if status != 'ok':
        _report_problem(status, res); _finish('evaluate', False); return
    ok = True
    mine = mean(run_episode(m, optimal_policy(m), i)[0][-1]['ret'] for i in range(100))
    if not _close(res.get('avg_return'), mine, 1e-9):
        print(f"❌ avg_return צריך להיות ממוצע התשואה של run_episode על הזרעים 0 עד 99 ({mine:.3f}), קיבלנו {res.get('avg_return')!r}.")
        ok = False
    elif abs(mine - REF_BOOK) > 0.08:
        print(f'❌ התשואה הממוצעת {mine:.3f} רחוקה מהערך הצפוי (בערך {REF_BOOK}). בדקו את sample ואת run_episode.'); ok = False
    if ok and sum(res.get('ends', {}).values()) != 100:
        print('❌ ends צריך לספור את כל 100 הריצות (לפי הערך של היציאה, או None אם נגמר הזמן).'); ok = False
    _finish('evaluate', ok)

def check_discount(best_fn, thr_fn):
    ok = True
    for g in (1.0, 0.5, 0.33, 0.3, 0.2, 0.1):
        exp = optimal_policy(GridMDP('row', gamma=g))[(3, 0)]
        status, got = _call(best_fn, (g,))
        if status != 'ok':
            _report_problem(status, got); ok = False; break
        if got != exp:
            print(f"❌ γ = {g}: מ-d המדיניות האופטימלית הולכת {'מערבה' if exp == 'W' else 'מזרחה'} ({exp!r}), קיבלנו {got!r}.")
            ok = False
    if ok:
        status, t = _call(thr_fn, ())
        if status != 'ok':
            _report_problem(status, t); ok = False
        elif not _close(t, 1 / math.sqrt(10), 1e-6):
            print(f'❌ נקודת השוויון צריכה להיות 1/√10 ≈ 0.316228, קיבלנו {t!r}.'); ok = False
        else:
            print(f'נקודת השוויון: {t:.6f}, ו-1/√10 = {1 / math.sqrt(10):.6f}')
    _finish('discount', ok)

def ready(*names):
    missing = [n for n in names if not RESULTS.get(n)]
    if missing:
        print('⚠️ השלימו קודם את המשימות האלה והריצו את תאי הבדיקה שלהן:', ', '.join(missing))
    return not missing

def final_report():
    for n in ['expectimax', 'outcomes', 'episodes', 'evaluate', 'discount']:
        mark = {True: '✅', False: '❌'}.get(RESULTS.get(n), '⬜ (לא נבדק)')
        print(f'{mark}  {n}')
'''


def md(text):
    return nbf.v4.new_markdown_cell('<div dir="rtl" align="right">\n\n' + text.strip() + '\n\n</div>')

def answer(key):
    return md('✍️ **תשובה (פתרון):**\n\n' + ANSWERS.get(key, '*(ימולא אחרי הרצת הפתרון)*'))

student_answer = lambda: md('✍️ **תשובה:** *(לחצו פעמיים על התא כדי לכתוב כאן)*')

# (student version, solution version) for each code cell with a TODO
TASK1 = ('''def expectimax(t, kind='max'):
    """ערך Expectimax של העץ t. kind = 'max' או 'chance' (סוג הצומת t). הרמות מתחלפות: max ← chance ← max ..."""
    # TODO: אם is_leaf(t): החזירו את t (התועלת של העלה)
    #       אם kind == 'max': החזירו את המקסימום של expectimax(child, 'chance') על כל הילדים
    #       אחרת (צומת מקריות): החזירו את הסכום של p * expectimax(child, 'max')
    #                            על כל הזוגות (p, child) ב-chance_children(t)
    pass
''', '''def expectimax(t, kind='max'):
    """ערך Expectimax של העץ t. kind = 'max' או 'chance' (סוג הצומת t). הרמות מתחלפות: max ← chance ← max ..."""
    if is_leaf(t):
        return t
    if kind == 'max':
        return max(expectimax(child, 'chance') for child in t)
    return sum(p * expectimax(child, 'max') for p, child in chance_children(t))
''')

TASK2 = ('''def outcomes(mdp, s, a):
    """
    כל התוצאות האפשריות של הפעולה a במשבצת s: רשימה של שלשות (p, d, next).
    p = ההסתברות, d = הכיוון שבו זזנו בפועל, next = המשבצת שהגענו אליה (None = הריצה נגמרה).
    """
    if a == 'X':                     # יציאה: מקבלים את התגמול של המשבצת, והריצה נגמרת
        return [(1.0, 'X', None)]
    result = []
    # TODO: שלוש אפשרויות (דלגו על אפשרות שההסתברות שלה 0):
    #   הכיוון שנבחר a, בהסתברות 1 - mdp.noise
    #   כל אחד משני הכיוונים שבצדדים, SIDE[a][0] ו-SIDE[a][1], בהסתברות mdp.noise / 2
    #   לכל כיוון d:  next = mdp.step_to(s, d)   (קיר או קצה הלוח: נשארים במקום)

    return result
''', '''def outcomes(mdp, s, a):
    """
    כל התוצאות האפשריות של הפעולה a במשבצת s: רשימה של שלשות (p, d, next).
    p = ההסתברות, d = הכיוון שבו זזנו בפועל, next = המשבצת שהגענו אליה (None = הריצה נגמרה).
    """
    if a == 'X':                     # יציאה: מקבלים את התגמול של המשבצת, והריצה נגמרת
        return [(1.0, 'X', None)]
    result = []
    for p, d in [(1 - mdp.noise, a), (mdp.noise / 2, SIDE[a][0]), (mdp.noise / 2, SIDE[a][1])]:
        if p > 0:
            result.append((p, d, mdp.step_to(s, d)))
    return result
''')

TASK3 = ('''def discounted_return(rewards, gamma):
    """r0 + γ·r1 + γ²·r2 + ...  (התשואה המהוונת של רצף תגמולים)"""
    total = 0.0
    # TODO: לכל t והתגמול r שלו (enumerate):  total += gamma ** t * r

    return total


def sample(mdp, s, a, rng):
    """מגרילים תוצאה אחת של הפעולה a במשבצת s. מחזירים (d, next)."""
    x = rng.random()                 # מספר אקראי בין 0 ל-1
    acc = 0.0
    # TODO: עברו על outcomes(mdp, s, a) לפי הסדר, וצברו את ההסתברויות ב-acc.
    #       ברגע ש-x < acc, החזירו (d, next) של התוצאה הנוכחית.

    return outcomes(mdp, s, a)[-1][1:]   # ליתר ביטחון (שגיאות עיגול)


def run_episode(mdp, policy, seed=0, max_steps=100):
    """ריצה אחת לפי המדיניות. מחזירה (frames, end): רשימת הצעדים, וערך היציאה שבה נגמרה הריצה (או None)."""
    rng = random.Random(seed)
    s, frames, rewards = mdp.start, [], []
    for t in range(max_steps):
        a = policy[s]
        # TODO: הגרילו (d, n) בעזרת sample, חשבו r = mdp.reward(s, a), והוסיפו את r ל-rewards

        frames.append(dict(t=t, s=s, a=a, d=d, n=n, r=r, ret=discounted_return(rewards, mdp.gamma)))
        # TODO: אם n is None (יצאנו): החזירו frames, mdp.exits[s]. אחרת: s = n

    return frames, None
''', '''def discounted_return(rewards, gamma):
    """r0 + γ·r1 + γ²·r2 + ...  (התשואה המהוונת של רצף תגמולים)"""
    total = 0.0
    for t, r in enumerate(rewards):
        total += gamma ** t * r
    return total


def sample(mdp, s, a, rng):
    """מגרילים תוצאה אחת של הפעולה a במשבצת s. מחזירים (d, next)."""
    x = rng.random()                 # מספר אקראי בין 0 ל-1
    acc = 0.0
    for p, d, nxt in outcomes(mdp, s, a):
        acc += p
        if x < acc:
            return d, nxt
    return outcomes(mdp, s, a)[-1][1:]   # ליתר ביטחון (שגיאות עיגול)


def run_episode(mdp, policy, seed=0, max_steps=100):
    """ריצה אחת לפי המדיניות. מחזירה (frames, end): רשימת הצעדים, וערך היציאה שבה נגמרה הריצה (או None)."""
    rng = random.Random(seed)
    s, frames, rewards = mdp.start, [], []
    for t in range(max_steps):
        a = policy[s]
        d, n = sample(mdp, s, a, rng)
        r = mdp.reward(s, a)
        rewards.append(r)
        frames.append(dict(t=t, s=s, a=a, d=d, n=n, r=r, ret=discounted_return(rewards, mdp.gamma)))
        if n is None:
            return frames, mdp.exits[s]
        s = n
    return frames, None
''')

TASK4 = ('''def evaluate(mdp, policy, n=100):
    """מריצים את המדיניות על הזרעים 0 עד n-1. מחזירים:
    avg_return = התשואה המהוונת הממוצעת, ends = {ערך היציאה: כמה ריצות נגמרו בה} (None = נגמר הזמן)."""
    returns, ends = [], {}
    # TODO: לכל seed בטווח n: frames, end = run_episode(mdp, policy, seed)
    #       הוסיפו את frames[-1]['ret'] ל-returns, והגדילו ב-1 את ends[end] (התחילו מ-0)

    return dict(avg_return=mean(returns) if returns else None, ends=ends)
''', '''def evaluate(mdp, policy, n=100):
    """מריצים את המדיניות על הזרעים 0 עד n-1. מחזירים:
    avg_return = התשואה המהוונת הממוצעת, ends = {ערך היציאה: כמה ריצות נגמרו בה} (None = נגמר הזמן)."""
    returns, ends = [], {}
    for seed in range(n):
        frames, end = run_episode(mdp, policy, seed)
        returns.append(frames[-1]['ret'])
        ends[end] = ends.get(end, 0) + 1
    return dict(avg_return=mean(returns) if returns else None, ends=ends)
''')

TASK5 = ('''def best_from_d(gamma):
    """בלוח row, בלי רעש: לאן כדאי ללכת מ-d? 'W' (מערבה, אל ה-10) או 'E' (מזרחה, אל ה-1)."""
    # TODO: מערבה: 3 צעדים ואז יציאה עם 10, כלומר 10 · γ³.  מזרחה: צעד אחד ואז יציאה עם 1, כלומר 1 · γ.
    #       החזירו 'W' אם הערך של מערבה גדול או שווה, ואחרת 'E'.
    pass


def threshold(iterations=60):
    """חיפוש בינארי: ה-γ שבו מזרחה ומערבה שווים (בין 0 ל-1)."""
    lo, hi = 0.0, 1.0
    # TODO: בכל איטרציה: mid = (lo + hi) / 2.
    #       אם best_from_d(mid) == 'W', נקודת השוויון משמאל ל-mid (hi = mid). אחרת היא מימין (lo = mid).

    return (lo + hi) / 2
''', '''def best_from_d(gamma):
    """בלוח row, בלי רעש: לאן כדאי ללכת מ-d? 'W' (מערבה, אל ה-10) או 'E' (מזרחה, אל ה-1)."""
    return 'W' if 10 * gamma ** 3 >= 1 * gamma else 'E'


def threshold(iterations=60):
    """חיפוש בינארי: ה-γ שבו מזרחה ומערבה שווים (בין 0 ל-1)."""
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if best_from_d(mid) == 'W':
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
''')


def guarded(code, *names):
    body = '\n'.join(('    ' + l) if l else l for l in code.split('\n'))
    return nbf.v4.new_code_cell(f"if ready({', '.join(map(repr, names))}):\n" + body)


def build(solution):
    c = lambda pair: nbf.v4.new_code_cell(pair[1] if solution else pair[0])
    ans = answer if solution else (lambda k: student_answer())
    code = nbf.v4.new_code_cell
    cells = [
        md('''
# HM_06: Expectimax ומבוא ל-MDP

**קורס: יסודות בינה מלאכותית · תרגול לשיעור 6 (אי-ודאות)**

בחלק הראשון של השיעור ראינו את **Expectimax**: במקום יריב שבוחר את הגרוע ביותר, יש **מקריות**, ומחשבים ממוצע משוקלל. בחלק השני ראינו **MDP**: סוכן בעולם שבו הפעולות **רועשות**, וכל צעד נותן **תגמול**. בתרגול הזה תממשו את שני הרעיונות, ותריצו אותם על **עולם הרשת** מההרצאה. זה אותו מנוע ואותו ציור כמו בסימולטור של הקורס (demos/mdp).

1. **Expectimax** על עצים מההרצאה, ולמה גודל הערכים חשוב.
2. **מודל המעברים** של עולם הרשת: 80% לכיוון שבחרנו, 10% לכל צד.
3. **ריצה ותשואה מהוונת:** מגרילים ריצה, ומחשבים את סכום התגמולים המהוון.
4. **השוואת מדיניות בסימולציה:** מחיר החיים.
5. **תרגיל ההיוון** מההרצאה, בקוד.

**איך עובדים:** הריצו קודם את תא ההגדרות. בכל ✏️ משימה השלימו את ה-`TODO`, והריצו את תא הבדיקה שאחריה. ב-❓ שאלות כתבו תשובה בתא ✍️ שאחריהן.
'''),
        code(SETUP),
        # ---------------- part 1 ----------------
        md('''
## חלק 1: Expectimax

עץ בתרגול הוא **רשימות מקוננות**: עלה הוא מספר (התועלת), וצומת פנימי הוא רשימה של ילדים. השורש הוא **MAX**, ומתחתיו **צמתי מקריות**, והרמות מתחלפות: MAX, מקריות, MAX, ...

בצומת מקריות, ילד יכול להופיע לבד (אז לכל הילדים אותה הסתברות), או כזוג `(p, child)` עם הסתברות. הפונקציה `chance_children(t)` מחזירה תמיד רשימה של זוגות `(p, child)`.

| מה | משמעות |
|---|---|
| `is_leaf(t)` | האם t הוא עלה (מספר) |
| `chance_children(t)` | הילדים של צומת מקריות עם ההסתברויות |
| `minimax(t)` | Minimax על אותו עץ (צמתי המקריות כ-MIN), להשוואה |
'''),
        code('''for name, (t, kind, value) in LECTURE_TREES.items():
    show_tree(t, kind, title=name)'''),
        md('''
### ✏️ משימה 1: Expectimax

ממשו את `expectimax` לפי הפסאודו-קוד מההרצאה:

<div dir="ltr">

```
def value(state):
    if the state is a terminal state: return the state's utility
    if the next agent is MAX: return max-value(state)
    if the next agent is EXP: return exp-value(state)

def exp-value(state):
    initialize v = 0
    for each successor of state:
        p = probability(successor)
        v += p * value(successor)
    return v
```

</div>
'''),
        c(TASK1),
        code('check_expectimax(expectimax)'),
        md('''
אחרי שהבדיקה עוברת, התא הבא מצייר את העצים מההרצאה **עם הערכים שהפונקציה שלכם מחשבת** בכל צומת. הקשת שהשורש בוחר מודגשת בירוק.
'''),
        guarded('''show_tree(*LECTURE_TREES['full example (3 12 9 | 2 4 6 | 15 6 0)'][:2], value_fn=expectimax, title='full example')
show_tree(*LECTURE_TREES['one chance node (8 24 -12, p = 1/2 1/3 1/6)'][:2], value_fn=expectimax, title='one chance node')
compare_trees([[10, 10], [9, 100]], 'worst case vs average case (lecture)')''', 'expectimax'),
        md('''
### ❓ שאלה 1: גודל הערכים, וגיזום

הריצו את התא הבא. הוא משווה את ההחלטה של Minimax ושל Expectimax על העץ מההרצאה (0, 40 | 20, 30), לפני ואחרי שמעלים כל עלה בריבוע.

1. מה בוחר כל אלגוריתם, לפני ואחרי? למה ההחלטה של Minimax לא השתנתה, ושל Expectimax כן? מה זה אומר על פונקציית הערכה ל-Expectimax?
2. באלפא-בטא גזמנו ענפים שלא יכולים לשנות את התוצאה. למה ב-Expectimax אי אפשר לגזום באותה צורה? מה היה משתנה אם היה ידוע מראש שכל העלים בין 0 ל-100?
'''),
        guarded('''t = [[0, 40], [20, 30]]
for name, tree in [('original', t), ('squared', square_leaves(t))]:
    compare_trees(tree, name)
    for algo, fn in [('minimax', minimax), ('expectimax', expectimax)]:
        i, vals = best_child(tree, fn)
        print(f"{name:9s} {algo:11s} children = {vals}  ->  {'left' if i == 0 else 'right'}")''', 'expectimax'),
        ans('q1'),
        # ---------------- part 2 ----------------
        md('''
## חלק 2: עולם הרשת ומודל המעברים

הלוח `book` הוא העולם מההרצאה: 4 × 3, קיר אחד, יציאה עם ‎+1 ויציאה עם ‎−1. משבצת היא זוג `(עמודה, שורה)`, והשורה 0 היא **התחתונה**.

| מה | משמעות |
|---|---|
| `mdp = GridMDP('book', noise=0.2, living=0.0, gamma=0.9)` | יצירת עולם. `living` הוא תגמול החיים R, ו-`gamma` הוא γ |
| `mdp.cells`, `mdp.walls`, `mdp.exits` | המשבצות, הקירות, והיציאות (מילון: משבצת ← תגמול) |
| `mdp.start` | משבצת ההתחלה |
| `mdp.actions(s)` | הפעולות במשבצת: `'N'`, `'S'`, `'E'`, `'W'`, ובמשבצת יציאה `'X'` |
| `mdp.step_to(s, d)` | לאן מגיעים מ-s בכיוון d (קיר או קצה: נשארים ב-s) |
| `mdp.reward(s, a)` | התגמול: ביציאה הערך של המשבצת, ובכל צעד אחר R |
| `SIDE[a]` | שני הכיוונים שבצדדים של a, למשל `SIDE['N'] == 'WE'` |
| `show_grid(mdp, policy)` | ציור הלוח, עם חיצים של מדיניות |
'''),
        code('''mdp = GridMDP('book')
print('start:', mdp.start, ' exits:', mdp.exits, ' walls:', mdp.walls)
show_grid(mdp, optimal_policy(mdp))'''),
        md('''
### ✏️ משימה 2: מודל המעברים T(s, a, s′)

עם רעש 0.2, הפעולה "צפונה" מצליחה ב-80% מהמקרים, וב-10% הסוכן זז מערבה וב-10% מזרחה. אם יש קיר בכיוון שבו זז, הוא נשאר במקום. ממשו את `outcomes`.
'''),
        c(TASK2),
        code('''check_outcomes(outcomes)
print(outcomes(GridMDP('book'), (0, 0), 'N'))   # צפונה מהפינה: 0.8 למעלה, 0.1 נתקע בקיר, 0.1 ימינה'''),
        md('''
### ❓ שאלה 2: מרקוב

1. במודל שלכם, ההסתברות להגיע ל-s′ תלויה רק ב-s וב-a. מה זה אומר "מרקובי"? מה **לא** משפיע על המעבר?
2. תנו דוגמה לעולם שבו ההנחה הזאת לא מתקיימת. למשל: הרובוט מחליק יותר אם הוא החליק גם בצעד הקודם. איך אפשר להפוך אותו שוב למרקובי, בעזרת שינוי של מה שנכלל ב"מצב"?
'''),
        ans('q2'),
        # ---------------- part 3 ----------------
        md('''
## חלק 3: ריצה ותשואה מהוונת

**מדיניות** (policy) היא מילון: לכל משבצת, איזו פעולה לעשות. **ריצה** (episode) מתחילה ב-`mdp.start`, ובכל צעד עושה את הפעולה של המדיניות, עד שיוצאים.

**התשואה המהוונת** של רצף תגמולים r₀, r₁, r₂, ... היא r₀ + γ·r₁ + γ²·r₂ + ... (בהרצאה: U([1, 2, 3]) = 2.75 כש-γ = 0.5).

### ✏️ משימה 3: ריצה

ממשו את שלוש הפונקציות. `sample` מגרילה **תוצאה אחת** מתוך `outcomes`: מגרילים מספר x בין 0 ל-1, וצוברים הסתברויות עד שהסכום עובר את x.
'''),
        c(TASK3),
        code('check_episodes(discounted_return, sample, run_episode)'),
        guarded('''mdp = GridMDP('book')
frames, end = run_episode(mdp, optimal_policy(mdp), seed=3)
print(f'end: {end:+g} after {len(frames) - 1} moves, return {frames[-1]["ret"]:.3f}')
animate(mdp, frames, optimal_policy(mdp))''', 'episodes'),
        # ---------------- part 4 ----------------
        md('''
## חלק 4: מחיר החיים

**מדיניות אחת מול אחרת:** הדרך היחידה שלנו (עד שיעור 7) להשוות מדיניות היא **סימולציה**: מריצים הרבה פעמים, ומחשבים ממוצע.

### ✏️ משימה 4: הערכה בסימולציה
'''),
        c(TASK4),
        code('check_evaluate(evaluate)'),
        md('''
עכשיו משווים שתי מדיניות, עם רעש 0.2 ובלי היוון (γ = 1), בארבעה ערכים של תגמול החיים R (כמו בשקף "מחיר החיים"):

- **optimal:** המדיניות האופטימלית לכל R (מוכנה מראש).
- **hand:** "הדרך הקצרה", מדיניות קבועה שכתבנו ביד: לאורך השורה התחתונה, ואז למעלה, צמוד לבור.
'''),
        guarded('''show_grid(GridMDP('book'), HAND)
print(f"{'R':>6} {'policy':>8} {'avg return':>11}  ends (200 runs)")
for R in [-0.01, -0.03, -0.4, -2.0]:
    mdp = GridMDP('book', noise=0.2, living=R, gamma=1.0)
    for name, pol in [('optimal', optimal_policy(mdp)), ('hand', HAND)]:
        res = evaluate(mdp, pol, 200)
        print(f"{R:>6} {name:>8} {res['avg_return']:>11.3f}  {res['ends']}")''', 'evaluate'),
        md('''
### ❓ שאלה 3: מחיר החיים

1. העתיקו את הטבלה.
2. עם R = ‎−0.01, למה המדיניות האופטימלית טובה יותר מ"הדרך הקצרה"? (רמז: מה קורה במשבצת שצמודה לבור, כשמחליקים?)
3. עם R = ‎−2, כמעט כל הריצות של המדיניות האופטימלית נגמרות ב-‎−1. למה זאת לא טעות?
4. למה הממוצע הוא רק **הערכה** לערך של המדיניות? מה היה קורה אם היינו מריצים 10 ריצות במקום 200?
'''),
        ans('q3'),
        # ---------------- part 5 ----------------
        md('''
## חלק 5: תרגיל ההיוון

הלוח `row` הוא התרגיל מההרצאה: חמש משבצות a, b, c, d, e בשורה, יציאה עם 10 מ-a ויציאה עם 1 מ-e, בלי רעש ובלי תגמול חיים. השאלה: מ-d, ללכת מערבה (אל ה-10, רחוק) או מזרחה (אל ה-1, קרוב)?
'''),
        code('''for g in [1.0, 0.1]:
    mdp = GridMDP('row', gamma=g)
    print('gamma =', g)
    show_grid(mdp, optimal_policy(mdp))'''),
        md('''
### ✏️ משימה 5: מתי מזרח ומערב שווים?
'''),
        c(TASK5),
        code('check_discount(best_from_d, threshold)'),
        md('''
### ❓ שאלה 4: היוון

1. כתבו את המשוואה שנקודת השוויון פותרת, ופתרו אותה ביד. האם יצא כמו בחיפוש הבינארי?
2. למה היוון גורם לסוכן להעדיף תגמול **קרוב**? תנו דוגמה מהחיים.
3. בעולם הרשת, מה יקרה עם γ = 1 ותגמול חיים **חיובי** (למשל ‎+1 לכל צעד)? מה תעשה המדיניות האופטימלית, ולמה צריך היוון (או סוף ידוע מראש) כדי שהתשואה תהיה סופית?
'''),
        ans('q4'),
        md('''
## סיכום והגשה

- **Expectimax** מחליף את צמתי ה-MIN ב**ממוצע משוקלל**. בגלל זה **גודל** הערכים חשוב, ולא רק הסדר, וגם גיזום כבר לא פשוט.
- **MDP** מתאר עולם עם מצבים, פעולות, **מודל מעברים** T, תגמולים R, והיוון γ. ההנחה המרקובית: המעבר תלוי רק במצב ובפעולה.
- **מדיניות** אומרת מה לעשות בכל מצב. את הערך שלה אפשר להעריך בסימולציה. בשיעור 7 נחשב אותו בדיוק, ונמצא את המדיניות האופטימלית בעצמנו (Value Iteration).
- **תגמול החיים** וה**היוון** משנים לגמרי את המדיניות האופטימלית.

הריצו את התא האחרון. לפני ההגשה ודאו שכל השורות מסומנות ✅, ושכל התשובות ✍️ מלאות.
'''),
        code('final_report()'),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {'colab': {'provenance': [], 'toc_visible': True},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                   'language_info': {'name': 'python'}}
    return nb


for sol, name in [(False, 'HM_06.ipynb'), (True, 'HM_06_solution.ipynb')]:
    nbf.write(build(sol), f'{OUT}/{name}')
print('written', OUT)
