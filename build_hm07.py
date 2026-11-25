"""Build HM_07.ipynb (student) and HM_07_solution.ipynb from one cell list (lesson 7: solving MDPs).
Usage: python build_hm07.py <engine mdp_engine.py> <out_dir> [answers.json] [--numbers]"""
import ast, json, sys
import nbformat as nbf

ENGINE_PATH, OUT = sys.argv[1], sys.argv[2]
ANSWERS = json.load(open(sys.argv[3], encoding='utf-8')) if len(sys.argv) > 3 and sys.argv[3] != '--numbers' else {}
sys.path.insert(0, ENGINE_PATH.rsplit('/', 1)[0])
import mdp_engine as me

# ---------- engine for the notebook: the transition model stays (HM_06), all planning goes (that is the exercise) ----
src = open(ENGINE_PATH, encoding='utf-8').read()
tree = ast.parse(src)
DROP = {'value_iteration', 'q_values', 'greedy_policy', 'policy_evaluation', 'policy_iteration', 'pi_stages'}
parts = []
for node in tree.body:
    if isinstance(node, (ast.If, ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and node is tree.body[0]):
        continue
    if getattr(node, 'name', None) in DROP:
        continue
    seg = ast.get_source_segment(src, node)
    if isinstance(node, ast.ClassDef):
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == 'q':
                seg = seg.replace('\n\n    ' + ast.get_source_segment(src, m), '')
    parts.append(seg)
engine = '\n\n'.join(parts)
assert 'def q(' not in engine and 'value_iteration' not in engine and 'def outcomes' in engine

# ---------- reference numbers (reference engine) ----------
r = lambda d: {k: round(v, 9) for k, v in d.items()}
BOOK = me.GridMDP('book')
VK = me.value_iteration(BOOK)
REF_VK = [r(V) for V in VK]                                  # V_0 .. V_32 on book
VSTAR = VK[-1]
REF_Q = {(s, a): round(BOOK.q(VSTAR, s, a), 9) for s in BOOK.cells for a in BOOK.actions(s)}
M2 = me.GridMDP('book', 0.2, -0.04, 0.5)
REF_Q2 = {(s, a): round(M2.q(VSTAR, s, a), 9) for s in M2.cells for a in M2.actions(s)}
REF_PSTAR = me.greedy_policy(BOOK, VSTAR)
LIFE = me.GridMDP('book', 0.2, -0.01, 1.0)
LIFE_V = r(me.value_iteration(LIFE)[-1])
LIFE_Q = {a: round(LIFE.q(LIFE_V, (2, 1), a), 9) for a in LIFE.actions((2, 1))}
EAST, NORTH = me.constant_policy(BOOK, 'E'), me.constant_policy(BOOK, 'N')
REF_VE, REF_VN = r(me.policy_evaluation(BOOK, EAST)[-1]), r(me.policy_evaluation(BOOK, NORTH)[-1])
CH = {a: [len(x['changed']) for x in me.pi_stages(BOOK, me.constant_policy(BOOK, a)) if x['kind'] == 'improve']
      for a in 'EN'}
OTHER = {}
for name in ('bridge', 'cliff', 'discount'):
    m = me.GridMDP(name)
    OTHER[name] = me.greedy_policy(m, me.value_iteration(m)[-1])
SIM = {n: round(me.run_many(BOOK, EAST, n)['avg_return'], 3) for n in (10, 100, 1000)}
assert round(REF_Q[((2, 2), 'E')], 2) == 0.85 and len(VK) - 1 == 32 and CH == {'E': [6, 2, 0], 'N': [5, 1, 0]}
if '--numbers' in sys.argv:
    print('V^E start', REF_VE[(0, 0)], 'sim', SIM, 'LIFE_Q', LIFE_Q, 'CH', CH)
    sys.exit()

SETUP = r'''#@title 🔧 תא הגדרות: הריצו אותו ראשון (אין צורך לקרוא את הקוד)
import json, math, random, signal
from statistics import mean
from IPython.display import HTML, display

# ============================================================
# מנוע "עולם הרשת" (אותו מנוע כמו בסימולטור של הקורס, demos/mdp)
# את outcomes כתבתם ב-HM_06, והיא כאן מוכנה (mdp.outcomes). כל החישוב של שיעור 7 נמחק: זה התרגיל.
# ============================================================
''' + engine + r'''


def show_grid(mdp, policy=None, values=None, q=None, changed=(), cell=80):
    """ציור הלוח. policy: {משבצת: פעולה}; values: {משבצת: מספר}; q: {משבצת: {פעולה: מספר}} (ארבעה משולשים);
    changed: משבצות שיסומנו במסגרת כתומה."""
    display(HTML(grid_svg(mdp, cell, V=values, Q=q, pol=policy, changed=changed)))

def side_by_side(*boards):
    """כמה לוחות בשורה אחת. כל לוח: (כותרת, mdp, policy, values)."""
    html = ''.join(f'<div style="margin:4px"><div style="font:600 15px sans-serif;margin-bottom:4px">{t}</div>'
                   f'{grid_svg(m, 70, V=v, pol=p, uid=f"sb{i}")}</div>' for i, (t, m, p, v) in enumerate(boards))
    display(HTML(f'<div style="display:flex;flex-wrap:wrap">{html}</div>'))

def player(pics, caps, uid='pl'):
    """נגן: ◀ ▶ בין תמונות (SVG) שחושבו מראש, עם כיתוב לכל תמונה."""
    btn = ('style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:3px 12px;'
           'cursor:pointer;font-size:15px;font-weight:600"')
    display(HTML(f"""<div style="background:{BG};color:{TXT};padding:10px;border-radius:12px;font-family:sans-serif">
<div style="display:flex;gap:6px;align-items:center;margin-bottom:8px">
<button id="{uid}b" {btn}>◀</button><button id="{uid}n" {btn}>▶</button><button id="{uid}e" {btn}>▶|</button>
<button id="{uid}r" {btn}>⟲</button><span id="{uid}c" style="font-size:15px;font-weight:600;margin-left:8px"></span></div>
<div id="{uid}v"></div></div>
<script>(function(){{
const P={json.dumps(pics)}, C={json.dumps(caps)}; let k=0;
const $=s=>document.getElementById("{uid}"+s);
function show(){{ $("v").innerHTML=P[k]; $("c").textContent=C[k]; }}
$("n").onclick=()=>{{ k=Math.min(P.length-1,k+1); show(); }};
$("b").onclick=()=>{{ k=Math.max(0,k-1); show(); }};
$("e").onclick=()=>{{ k=P.length-1; show(); }};
$("r").onclick=()=>{{ k=0; show(); }};
show(); }})();</script>"""))

def changed_values(V, prev, tol=0.005):
    return [s for s in V if prev is not None and abs(V[s] - prev[s]) > tol]

def values_player(mdp, Vs, uid='vk'):
    """נגן של V_0, V_1, ...: משבצות שהערך שלהן השתנה (ביותר מ-0.005) מסומנות בכתום."""
    pics, caps = [], []
    for k, V in enumerate(Vs):
        prev = Vs[k - 1] if k else None
        pics.append(grid_svg(mdp, 80, V=V, changed=changed_values(V, prev), uid=f'{uid}{k}'))
        ch = max(abs(V[s] - prev[s]) for s in V) if prev else 0
        caps.append(f'k = {k}' + (f' · largest change {ch:.4f}' if k else ''))
    player(pics, caps, uid)

def policy_name(policy):
    return ' '.join(f'{s}:{a}' for s, a in sorted(policy.items()))

# ============================================================
# מספרי ייחוס לבדיקות (חושבו מראש במנוע של הקורס)
# ============================================================
REF_VK = ''' + repr(REF_VK) + r'''
VSTAR = REF_VK[-1]
REF_Q = ''' + repr(REF_Q) + r'''
REF_Q2 = ''' + repr(REF_Q2) + r'''
REF_PSTAR = ''' + repr(REF_PSTAR) + r'''
LIFE_V = ''' + repr(LIFE_V) + r'''
REF_VE = ''' + repr(REF_VE) + r'''
REF_VN = ''' + repr(REF_VN) + r'''
REF_CH = ''' + repr(CH) + r'''
REF_OTHER = ''' + repr(OTHER) + r'''

# ============================================================
# תשתית בדיקות
# ============================================================
RESULTS = {}

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
        print('❌ הפונקציה לא הסתיימה בזמן: כנראה לולאה שלא נגמרת (האם תנאי העצירה נבדק?).')
    else:
        print(f'❌ הפונקציה זרקה שגיאה: {type(value).__name__}: {value}')
        print('   אם עדיין לא השלמתם את ה-TODO, זה צפוי. אחרת, קראו את הודעת השגיאה ובדקו את השורה הרלוונטית.')

def _finish(name, ok):
    RESULTS[name] = ok
    print(f'✅ {name}: כל הבדיקות עברו!' if ok else f'⛔ {name}: יש בדיקות שנכשלו. תקנו והריצו שוב.')

def _close(a, b, tol=1e-6):
    return isinstance(a, (int, float)) and abs(a - b) <= tol

def _vdiff(got, exp, tol=1e-6):
    """המשבצת הראשונה שבה V שונה מהצפוי (או None)."""
    if not isinstance(got, dict):
        return 'not a dict'
    for s, v in exp.items():
        if not _close(got.get(s), v, tol):
            return s
    return None

def check_q_value(fn):
    ok = True
    for mdp, V, ref, label in [(GridMDP('book'), VSTAR, REF_Q, 'book, V*'),
                               (GridMDP('book', 0.2, -0.04, 0.5), VSTAR, REF_Q2, 'R = −0.04, γ = 0.5')]:
        for (s, a), exp in ref.items():
            status, got = _call(fn, (mdp, V, s, a))
            if status != 'ok':
                _report_problem(status, got); _finish('q_value', False); return
            if not _close(got, exp):
                print(f'❌ [{label}] q_value ב-{s} עם {a!r}: קיבלנו {got!r}, צריך {exp:.4f}.')
                if a == 'X':
                    print('   רמז: ביציאה יש תוצאה אחת (next = None): התגמול של היציאה, בלי המשך.')
                else:
                    print('   רמז: סכום על outcomes של p · (R + γ · V[next]). R = mdp.reward(s, a), γ = mdp.gamma.')
                ok = False; break
        if not ok:
            break
    _finish('q_value', ok)

def check_value_iteration(fn):
    m = GridMDP('book')
    status, Vs = _call(fn, (m, 3))
    if status != 'ok':
        _report_problem(status, Vs); _finish('value_iteration', False); return
    ok = True
    if not isinstance(Vs, list) or len(Vs) != 4:
        print(f'❌ value_iteration(mdp, 3) צריכה להחזיר רשימה של 4 מילונים: V_0, V_1, V_2, V_3. קיבלנו {type(Vs).__name__}.')
        _finish('value_iteration', False); return
    for k in range(4):
        bad = _vdiff(Vs[k], REF_VK[k])
        if bad:
            print(f'❌ V_{k} שונה מהצפוי במשבצת {bad}: קיבלנו {Vs[k].get(bad) if isinstance(Vs[k], dict) else Vs[k]!r}, '
                  f'צריך {REF_VK[k][bad] if bad in REF_VK[k] else "?"}.')
            if k == 0:
                print('   רמז: V_0 = 0 בכל משבצת (כולל היציאות).')
            else:
                print('   רמז: V_{k+1}(s) = max על הפעולות של q_value(mdp, V_k, s, a). חשבו את כל V_{k+1} מתוך V_k הישן.')
            ok = False; break
    if ok:
        status, Vs = _call(fn, (m,))
        if status != 'ok':
            _report_problem(status, Vs); ok = False
        elif len(Vs) - 1 != len(REF_VK) - 1 or _vdiff(Vs[-1], VSTAR):
            print(f'❌ בלי k, צריך להמשיך עד שהשינוי הגדול ביותר קטן מ-tol: {len(REF_VK) - 1} סבבים, '
                  f'וערכים כמו בשקף. קיבלנו {len(Vs) - 1} סבבים.'); ok = False
    _finish('value_iteration', ok)

def check_extract(ext_fn, q_fn):
    m = GridMDP('book')
    status, pol = _call(ext_fn, (m, VSTAR))
    if status != 'ok':
        _report_problem(status, pol); _finish('extract', False); return
    ok = True
    if not isinstance(pol, dict) or any(pol.get(s) != a for s, a in REF_PSTAR.items()):
        bad = [s for s, a in REF_PSTAR.items() if not isinstance(pol, dict) or pol.get(s) != a]
        print(f'❌ על V* של book, המדיניות שונה מהאופטימלית במשבצות {bad[:4]}.')
        print("   רמז: argmax של q_value על mdp.actions(s). במשבצת יציאה הפעולה היחידה היא 'X'."); ok = False
    if ok:
        life = GridMDP('book', 0.2, -0.01, 1.0)
        p2 = ext_fn(life, LIFE_V)
        if p2.get((2, 1)) != 'W':
            print(f"❌ בהגדרה 'מחיר החיים', ליד הבור (2, 1) צריך להיות 'W' (מערבה), קיבלנו {p2.get((2, 1))!r}."); ok = False
    if ok:
        Q = {s: {a: q_value(m, VSTAR, s, a) for a in m.actions(s)} for s in m.cells}
        status, p3 = _call(q_fn, (Q,))
        if status != 'ok':
            _report_problem(status, p3); ok = False
        elif p3 != pol:
            print('❌ policy_from_q צריכה לתת את אותה מדיניות כמו extract_policy על אותם ערכים.'); ok = False
    _finish('extract', ok)

def check_evaluate(fn):
    m = GridMDP('book')
    ok = True
    for name, pol, ref in [('always East', constant_policy(m, 'E'), REF_VE), ('always North', constant_policy(m, 'N'), REF_VN),
                           ('optimal', REF_PSTAR, VSTAR)]:
        status, Vs = _call(fn, (m, pol))
        if status != 'ok':
            _report_problem(status, Vs); ok = False; break
        V = Vs[-1] if isinstance(Vs, list) and Vs else Vs
        bad = _vdiff(V, ref)
        if bad:
            print(f'❌ [{name}] במשבצת {bad}: קיבלנו {V.get(bad) if isinstance(V, dict) else V!r}, צריך {ref[bad]:.4f}.')
            print('   רמז: כמו value_iteration, אבל בלי max: רק q_value של הפעולה policy[s].'); ok = False; break
    _finish('evaluate_policy', ok)

def check_policy_iteration(fn):
    ok = True
    for start, exp in [('E', REF_CH['E']), ('N', REF_CH['N'])]:
        m = GridMDP('book')
        status, pols = _call(fn, (m, constant_policy(m, start)), seconds=60)
        if status != 'ok':
            _report_problem(status, pols); ok = False; break
        if not isinstance(pols, list) or len(pols) < 2:
            print('❌ policy_iteration צריכה להחזיר רשימה של מדיניות: π_0 (ההתחלה), π_1, ... עד הסופית.'); ok = False; break
        ch = [sum(1 for s in m.cells if b[s] != a[s]) for a, b in zip(pols, pols[1:])]
        if pols[-1] != REF_PSTAR:
            print(f"❌ מ-'always {start}' המדיניות הסופית לא אופטימלית."); ok = False; break
        if ch + [0] != exp and ch != exp:
            print(f"❌ מ-'always {start}': מספר החצים שהשתנו בכל שיפור {ch}, צריך {exp[:-1]}.")
            print('   רמז: בשוויון בין הפעולה הנוכחית לטובה ביותר, השאירו את הנוכחית.'); ok = False; break
    if ok:
        for name, ref in REF_OTHER.items():
            m = GridMDP(name)
            status, pols = _call(fn, (m, constant_policy(m, 'E')), seconds=60)
            if status != 'ok' or pols[-1] != ref:
                print(f'❌ על הלוח {name} המדיניות הסופית שונה מזו של Value Iteration.'); ok = False; break
    _finish('policy_iteration', ok)

def check_race(T):
    ok = True
    try:
        Vs = dict_value_iteration(T, 1.0, 3)
        want = [{'cool': 0, 'warm': 0, 'over': 0}, {'cool': 2, 'warm': 1, 'over': 0},
                {'cool': 3.5, 'warm': 2.5, 'over': 0}, {'cool': 5, 'warm': 4, 'over': 0}]
        for k, w in enumerate(want):
            if any(not _close(Vs[k][s], v) for s, v in w.items()):
                print(f'❌ בלי היוון, V_{k} צריך להיות {w}, קיבלנו {Vs[k]}.'); ok = False; break
        if ok:
            V = dict_value_iteration(T, 0.9)[-1]
            if not (_close(V['cool'], 15.5, 1e-4) and _close(V['warm'], 14.5, 1e-4)):
                print(f'❌ עם γ = 0.9 הערכים צריכים להתכנס ל-cool 15.5 ו-warm 14.5. קיבלנו {V}.'); ok = False
    except Exception as e:
        _report_problem('error', e); ok = False
    _finish('race (bonus)', ok)

def ready(*names):
    missing = [n for n in names if not RESULTS.get(n)]
    if missing:
        print('⚠️ השלימו קודם את המשימות האלה והריצו את תאי הבדיקה שלהן:', ', '.join(missing))
    return not missing

def final_report():
    for n in ['q_value', 'value_iteration', 'extract', 'evaluate_policy', 'policy_iteration', 'race (bonus)']:
        mark = {True: '✅', False: '❌'}.get(RESULTS.get(n), '⬜ (לא נבדק)')
        print(f'{mark}  {n}')
'''


def md(text):
    return nbf.v4.new_markdown_cell('<div dir="rtl" align="right">\n\n' + text.strip() + '\n\n</div>')

def answer(key):
    return md('✍️ **תשובה (פתרון):**\n\n' + ANSWERS.get(key, '*(ימולא אחרי הרצת הפתרון)*'))

student_answer = lambda: md('✍️ **תשובה:** *(לחצו פעמיים על התא כדי לכתוב כאן)*')

# (student version, solution version) for each code cell with a TODO
TASK1 = ('''def q_value(mdp, V, s, a):
    """ערך Q של הפעולה a במשבצת s, כשהעתיד נמדד לפי V:  Σ p · (R + γ · V[next]).
    mdp.outcomes(s, a) מחזירה שלשות (p, d, next). next = None: הריצה נגמרה (יציאה), ואין המשך."""
    R = mdp.reward(s, a)
    total = 0.0
    # TODO: לכל (p, d, nxt) ב-mdp.outcomes(s, a):
    #       העתיד הוא mdp.gamma * V[nxt], או 0 אם nxt is None.  total += p * (R + העתיד)

    return total
''', '''def q_value(mdp, V, s, a):
    """ערך Q של הפעולה a במשבצת s, כשהעתיד נמדד לפי V:  Σ p · (R + γ · V[next]).
    mdp.outcomes(s, a) מחזירה שלשות (p, d, next). next = None: הריצה נגמרה (יציאה), ואין המשך."""
    R = mdp.reward(s, a)
    total = 0.0
    for p, d, nxt in mdp.outcomes(s, a):
        future = mdp.gamma * V[nxt] if nxt is not None else 0.0
        total += p * (R + future)
    return total
''')

TASK2 = ('''def value_iteration(mdp, k=None, tol=1e-9, k_max=1000):
    """מחזירה רשימה [V_0, V_1, ..., V_k]. כל V הוא מילון {משבצת: ערך}.
    בלי k: ממשיכים עד שהשינוי הגדול ביותר בין V_k ל-V_{k+1} קטן מ-tol (ולכל היותר k_max סבבים)."""
    Vs = [{s: 0.0 for s in mdp.cells}]
    for _ in range(k if k is not None else k_max):
        V = Vs[-1]
        # TODO: new = {s: המקסימום של q_value(mdp, V, s, a) על כל a ב-mdp.actions(s)}  לכל s ב-mdp.cells
        #       (חשוב: מחשבים את כל new מתוך V הישן. לא מעדכנים את V תוך כדי.)
        new = V
        Vs.append(new)
        if k is None and max(abs(new[s] - V[s]) for s in mdp.cells) < tol:
            break
    return Vs
''', '''def value_iteration(mdp, k=None, tol=1e-9, k_max=1000):
    """מחזירה רשימה [V_0, V_1, ..., V_k]. כל V הוא מילון {משבצת: ערך}.
    בלי k: ממשיכים עד שהשינוי הגדול ביותר בין V_k ל-V_{k+1} קטן מ-tol (ולכל היותר k_max סבבים)."""
    Vs = [{s: 0.0 for s in mdp.cells}]
    for _ in range(k if k is not None else k_max):
        V = Vs[-1]
        new = {s: max(q_value(mdp, V, s, a) for a in mdp.actions(s)) for s in mdp.cells}
        Vs.append(new)
        if k is None and max(abs(new[s] - V[s]) for s in mdp.cells) < tol:
            break
    return Vs
''')

TASK3 = ('''def extract_policy(mdp, V):
    """חילוץ מדיניות: בכל משבצת, הפעולה עם q_value הגבוה ביותר לפי V (argmax)."""
    policy = {}
    for s in mdp.cells:
        # TODO: policy[s] = הפעולה a ב-mdp.actions(s) שנותנת את q_value(mdp, V, s, a) הגבוה ביותר
        #       (רמז: max(..., key=lambda a: ...))
        pass
    return policy


def policy_from_q(Q):
    """אותו דבר, מתוך ערכי Q: Q[s] הוא מילון {פעולה: ערך}."""
    # TODO: שורה אחת: לכל s, הפעולה עם הערך הגבוה ביותר ב-Q[s]
    pass
''', '''def extract_policy(mdp, V):
    """חילוץ מדיניות: בכל משבצת, הפעולה עם q_value הגבוה ביותר לפי V (argmax)."""
    policy = {}
    for s in mdp.cells:
        policy[s] = max(mdp.actions(s), key=lambda a: q_value(mdp, V, s, a))
    return policy


def policy_from_q(Q):
    """אותו דבר, מתוך ערכי Q: Q[s] הוא מילון {פעולה: ערך}."""
    return {s: max(Q[s], key=Q[s].get) for s in Q}
''')

TASK4 = ('''def evaluate_policy(mdp, policy, tol=1e-9, k_max=1000):
    """הערכת מדיניות: מחזירה [V_0, V_1, ...] של המדיניות policy. כמו value_iteration, אבל בלי max:
    בכל משבצת משתמשים רק בפעולה policy[s]."""
    Vs = [{s: 0.0 for s in mdp.cells}]
    for _ in range(k_max):
        V = Vs[-1]
        # TODO: new = {s: q_value(mdp, V, s, policy[s])}  לכל s ב-mdp.cells
        new = V
        Vs.append(new)
        if max(abs(new[s] - V[s]) for s in mdp.cells) < tol:
            break
    return Vs
''', '''def evaluate_policy(mdp, policy, tol=1e-9, k_max=1000):
    """הערכת מדיניות: מחזירה [V_0, V_1, ...] של המדיניות policy. כמו value_iteration, אבל בלי max:
    בכל משבצת משתמשים רק בפעולה policy[s]."""
    Vs = [{s: 0.0 for s in mdp.cells}]
    for _ in range(k_max):
        V = Vs[-1]
        new = {s: q_value(mdp, V, s, policy[s]) for s in mdp.cells}
        Vs.append(new)
        if max(abs(new[s] - V[s]) for s in mdp.cells) < tol:
            break
    return Vs
''')

TASK5 = ('''def policy_iteration(mdp, policy):
    """Policy Iteration. מחזירה את רשימת המדיניות [π_0, π_1, ...]: ההתחלה, ואחריה כל שיפור, עד שאף חץ לא משתנה."""
    policies = [dict(policy)]
    while True:
        V = evaluate_policy(mdp, policies[-1])[-1]           # צעד 1: הערכה
        best = extract_policy(mdp, V)                        # צעד 2: שיפור (חילוץ מדיניות מתוך V)
        new = {}
        for s in mdp.cells:
            # TODO: אם q_value של best[s] גדול ממש (ביותר מ-1e-9) מ-q_value של הפעולה הנוכחית, new[s] = best[s].
            #       אחרת (שוויון): new[s] = הפעולה הנוכחית policies[-1][s]
            new[s] = policies[-1][s]
        # TODO: אם new זהה ל-policies[-1]: החזירו את policies. אחרת: הוסיפו את new לרשימה
        return policies
''', '''def policy_iteration(mdp, policy):
    """Policy Iteration. מחזירה את רשימת המדיניות [π_0, π_1, ...]: ההתחלה, ואחריה כל שיפור, עד שאף חץ לא משתנה."""
    policies = [dict(policy)]
    while True:
        V = evaluate_policy(mdp, policies[-1])[-1]           # צעד 1: הערכה
        best = extract_policy(mdp, V)                        # צעד 2: שיפור (חילוץ מדיניות מתוך V)
        new = {}
        for s in mdp.cells:
            cur = policies[-1][s]
            if q_value(mdp, V, s, best[s]) > q_value(mdp, V, s, cur) + 1e-9:
                new[s] = best[s]
            else:
                new[s] = cur
        if new == policies[-1]:
            return policies
        policies.append(new)
''')

TASK6 = ('''# ה-MDP של המכונית משיעור 6. מצבים: 'cool' (קר), 'warm' (חם), 'over' (התחממות יתר, סוף).
# RACE[(מצב, פעולה)] = רשימה של (הסתברות, מצב הבא, תגמול)
RACE = {
    ('cool', 'slow'): [(1.0, 'cool', 1)],
    # TODO: ('cool', 'fast'): חצי מהזמן נשארים קרים, חצי מהזמן מתחממים. תגמול 2 בכל מקרה.
    # TODO: ('warm', 'slow'): חצי קר, חצי חם. תגמול 1.
    # TODO: ('warm', 'fast'): תמיד התחממות יתר. תגמול -10.
}
''', '''# ה-MDP של המכונית משיעור 6. מצבים: 'cool' (קר), 'warm' (חם), 'over' (התחממות יתר, סוף).
# RACE[(מצב, פעולה)] = רשימה של (הסתברות, מצב הבא, תגמול)
RACE = {
    ('cool', 'slow'): [(1.0, 'cool', 1)],
    ('cool', 'fast'): [(0.5, 'cool', 2), (0.5, 'warm', 2)],
    ('warm', 'slow'): [(0.5, 'cool', 1), (0.5, 'warm', 1)],
    ('warm', 'fast'): [(1.0, 'over', -10)],
}
''')

DICT_VI = '''def dict_value_iteration(T, gamma, k=None, tol=1e-9, k_max=1000):
    """Value Iteration על MDP שמוגדר כמילון T[(state, action)] = [(p, next, reward), ...] (מוכנה, אין מה להשלים).
    מצב בלי פעולות (כמו 'over') הוא מצב סיום, וערכו 0."""
    states = {s for s, _ in T} | {n for outs in T.values() for _, n, _ in outs}
    Vs = [{s: 0.0 for s in states}]
    for _ in range(k if k is not None else k_max):
        V = Vs[-1]
        new = {s: max((sum(p * (r + gamma * V[n]) for p, n, r in T[(s, a)]) for (s2, a) in T if s2 == s), default=0.0)
               for s in states}
        Vs.append(new)
        if k is None and max(abs(new[s] - V[s]) for s in states) < tol:
            break
    return Vs'''


def guarded(code, *names):
    body = '\n'.join(('    ' + l) if l else l for l in code.split('\n'))
    return nbf.v4.new_code_cell(f"if ready({', '.join(map(repr, names))}):\n" + body)


def build(solution):
    c = lambda pair: nbf.v4.new_code_cell(pair[1] if solution else pair[0])
    ans = answer if solution else (lambda k: student_answer())
    code = nbf.v4.new_code_cell
    cells = [
        md('''
# HM_07: פתרון MDP: ‏Value Iteration ו-Policy Iteration

ב-HM_06 הערכנו מדיניות **בסימולציה**, וקיבלנו את המדיניות האופטימלית כ"קופסה שחורה". בתרגול הזה נחשב אותן **בעצמנו**, כמו בשיעור 7:

1. **צעד אחד של בלמן:** ערך Q של פעולה, מתוך ערכים של המצבים הבאים.
2. **Value Iteration:** הערכים V₀, V₁, V₂, … עד שהם מתכנסים.
3. **חילוץ מדיניות:** מערכים לפעולות, מתוך V ומתוך Q.
4. **הערכת מדיניות:** כמה שווה מדיניות נתונה, בדיוק ולא בסימולציה.
5. **Policy Iteration:** הערכה ושיפור לסירוגין.
6. ✨ **בונוס:** המכונית משיעור 6, כ-MDP קטן.

**איך עובדים:** הריצו קודם את תא ההגדרות. בכל ✏️ משימה השלימו את ה-`TODO`, והריצו את תא הבדיקה שאחריה. ב-❓ שאלות כתבו תשובה בתא ✍️ שאחריהן.

**סימונים:** s משבצת, a פעולה, s′ המשבצת הבאה, p = T(s, a, s′) ההסתברות לעבור מ-s ל-s′ בפעולה a, ‏R התגמול על הצעד (`mdp.reward(s, a)`), ‏γ מקדם ההיוון (`mdp.gamma`).
'''),
        code(SETUP),
        # ---------------- part 1 ----------------
        md('''
## חלק 1: צעד אחד של בלמן

**ערך Q** של פעולה a במשבצת s הוא התועלת הצפויה אם עושים עכשיו את a, וממשיכים לפי הערכים V:

<div dir="ltr">

Q(s, a) = Σ<sub>s′</sub> T(s, a, s′) · [ R + γ · V(s′) ]

</div>

ב-HM_06 כתבתם את `outcomes`. היא מוכנה כאן כ-`mdp.outcomes(s, a)`, ומחזירה שלשות `(p, d, next)`. כש-`next` הוא `None` (יציאה), אין המשך: מקבלים רק את התגמול.

### ✏️ משימה 1: ‏q_value
'''),
        c(TASK1),
        code('check_q_value(q_value)'),
        md('''
התא הבא מצייר את הלוח עם **ערכי ה-Q שהפונקציה שלכם מחשבת** מתוך V\\*: בכל משבצת ארבעה משולשים, אחד לכל פעולה. הגבוה ביותר מודגש. במשבצת שמשמאל ל-‎+1, "מזרח" צריך לצאת 0.85 = 0.72 + 0.08 + 0.05, כמו בשקף.
'''),
        guarded('''mdp = GridMDP('book')
Q = {s: {a: q_value(mdp, VSTAR, s, a) for a in mdp.actions(s)} for s in mdp.cells if s not in mdp.exits}
show_grid(mdp, values=VSTAR, q=Q)
print('left of +1, East:', round(q_value(mdp, VSTAR, (2, 2), 'E'), 4))''', 'q_value'),
        # ---------------- part 2 ----------------
        md('''
## חלק 2: ‏Value Iteration

**ערכים מוגבלי זמן:** ‏V<sub>k</sub>(s) הוא הערך הטוב ביותר שאפשר להשיג מ-s, אם המשחק נגמר בעוד k צעדים. מתחילים מ-V₀ = 0, ובכל סבב:

<div dir="ltr">

V<sub>k+1</sub>(s) = max<sub>a</sub> Q<sub>k</sub>(s, a) = max<sub>a</sub> Σ<sub>s′</sub> T(s, a, s′) · [ R + γ · V<sub>k</sub>(s′) ]

</div>

### ✏️ משימה 2: ‏value_iteration
'''),
        c(TASK2),
        code('check_value_iteration(value_iteration)'),
        md('''
הנגן מראה את V₀, V₁, … עד ההתכנסות. משבצות שהערך שלהן השתנה מסומנות בכתום. שימו לב ל-0.72 שמופיע ב-V₂, ולערכים ש"מתפשטים" מה-‎+1.
'''),
        guarded('''mdp = GridMDP('book')
Vs = value_iteration(mdp)
print('converged after', len(Vs) - 1, 'sweeps')
values_player(mdp, Vs)''', 'value_iteration'),
        # ---------------- part 3 ----------------
        md('''
## חלק 3: חילוץ מדיניות

**חילוץ מדיניות** (Policy Extraction): נתונים ערכים V. בכל משבצת בוחרים את הפעולה עם ערך ה-Q הגבוה ביותר. זה **argmax**: לא הערך הגבוה ביותר, אלא **הפעולה** שנותנת אותו.

### ✏️ משימה 3: ‏extract_policy ו-policy_from_q
'''),
        c(TASK3),
        code('check_extract(extract_policy, policy_from_q)'),
        md('''
**כמה מתייצבת המדיניות?** התא הבא מחלץ מדיניות מכל V<sub>k</sub>, ומוצא מאיזה k החצים כבר לא משתנים. אחר כך הוא מצייר את ההגדרה "מחיר החיים" משיעור 6 (רעש 0.2, ‏γ = 1, ‏R = ‎−0.01), עם ערכי ה-Q במשבצת שליד הבור.
'''),
        guarded('''mdp = GridMDP('book')
Vs = value_iteration(mdp)
pols = [extract_policy(mdp, V) for V in Vs]
stable = next(k for k in range(1, len(pols)) if all(p == pols[-1] for p in pols[k:]))
print(f'policy final from k = {stable}, values converged at k = {len(Vs) - 1}')
show_grid(mdp, pols[-1], Vs[-1])
life = GridMDP('book', 0.2, -0.01, 1.0)
V = value_iteration(life)[-1]
near = (2, 1)
print('near the pit:', {a: round(q_value(life, V, near, a), 2) for a in life.actions(near)})
show_grid(life, extract_policy(life, V), V, q={near: {a: q_value(life, V, near, a) for a in life.actions(near)}},
          changed=[near])''', 'value_iteration', 'extract'),
        md('''
### ❓ שאלה 1: מדיניות מול ערכים

1. מאיזה k המדיניות כבר לא משתנה, ומתי הערכים מתכנסים? למה המדיניות מתייצבת מוקדם כל כך?
2. מה זה אומר על העבודה ש-Value Iteration עושה בסבבים האחרונים?
'''),
        ans('q1'),
        md('''
### ❓ שאלה 2: הריקוד ליד הבור

1. בהגדרה "מחיר החיים", המשבצת שליד הבור שווה 0.89, והמשבצת שמעליה 0.98. למה הסוכן הולך **מערבה**, אל הקיר, ולא צפונה? (היעזרו בערכי ה-Q שהודפסו.)
2. למה `policy_from_q` היא שורה אחת, ו-`extract_policy` צריכה לחשב צעד של Expectimax? מה עדיף לשמור אם רוצים לבחור פעולות מהר?
'''),
        ans('q2'),
        # ---------------- part 4 ----------------
        md('''
## חלק 4: הערכת מדיניות

**הערכת מדיניות** (Policy Evaluation): נותנים לנו מדיניות π, ומחשבים את V<sup>π</sup>, הערך של כל משבצת אם פועלים לפי π. אין max: בכל משבצת יש רק פעולה אחת.

<div dir="ltr">

V<sup>π</sup>(s) = Σ<sub>s′</sub> T(s, π(s), s′) · [ R + γ · V<sup>π</sup>(s′) ]

</div>

### ✏️ משימה 4: ‏evaluate_policy
'''),
        c(TASK4),
        code('check_evaluate(evaluate_policy)'),
        md('''
**חישוב מול סימולציה:** ב-HM_06 העריכו מדיניות בעזרת ממוצע של ריצות. כאן משווים את הממוצע (מ-`run_many`, אותה פונקציה כמו בסימולטור) לערך המדויק במשבצת ההתחלה, עבור "תמיד מזרחה".
'''),
        guarded('''mdp = GridMDP('book')
east = constant_policy(mdp, 'E')
VE = evaluate_policy(mdp, east)[-1]
side_by_side(('always East', mdp, east, VE), ('optimal π*', mdp, REF_PSTAR, VSTAR))
print(f'exact value at the start (0, 0): {VE[(0, 0)]:.3f}')
for n in [10, 100, 1000]:
    print(f'simulation, {n:4d} episodes: {run_many(mdp, east, n)["avg_return"]:.3f}')''', 'evaluate_policy'),
        # ---------------- part 5 ----------------
        md('''
## חלק 5: ‏Policy Iteration

**איטרציית מדיניות** (Policy Iteration): מתחילים ממדיניות כלשהי, ומשפרים אותה:
1. **הערכה:** מחשבים את V<sup>π</sup> של המדיניות הנוכחית (בלי max).
2. **שיפור:** חילוץ מדיניות מתוך V<sup>π</sup>.
3. חוזרים עד שאף חץ לא משתנה.

**שימו לב:** אם הפעולה הנוכחית טובה בדיוק כמו הטובה ביותר (שוויון), משאירים אותה. אחרת הלולאה יכולה להתחלף בין שתי פעולות שוות, ולא להיעצר.

### ✏️ משימה 5: ‏policy_iteration
'''),
        c(TASK5),
        code('check_policy_iteration(policy_iteration)'),
        md('''
הנגן מראה את כל השלבים מ"תמיד מזרחה": הערכה, שיפור, הערכה, ... חצים או ערכים שהשתנו מסומנים בכתום.
'''),
        guarded('''mdp = GridMDP('book')
pols = policy_iteration(mdp, constant_policy(mdp, 'E'))
pics, caps, prevV = [], [], None
for i, pol in enumerate(pols):
    Vs = evaluate_policy(mdp, pol)
    V = Vs[-1]
    if i:
        ch = [s for s in mdp.cells if pol[s] != pols[i - 1][s]]
        pics.append(grid_svg(mdp, 80, V=prevV, pol=pol, changed=ch, uid=f'pi{i}a'))
        caps.append(f'improvement {i}: {len(ch)} arrows changed')
    pics.append(grid_svg(mdp, 80, V=V, pol=pol, changed=changed_values(V, prevV), uid=f'pi{i}b'))
    caps.append(f'evaluation {i + 1}: {len(Vs) - 1} sweeps (no max)')
    prevV = V
pics.append(grid_svg(mdp, 80, V=prevV, pol=pols[-1], uid='piend')); caps.append(f'improvement {len(pols)}: no arrow changed → π*')
player(pics, caps, 'pi')
print('policy iteration == value iteration:', pols[-1] == extract_policy(mdp, value_iteration(mdp)[-1]))''',
                'policy_iteration', 'value_iteration'),
        md('''
### ❓ שאלה 3: ‏Value Iteration מול Policy Iteration

1. כמה סבבים עשה Value Iteration עד שהמדיניות התייצבה, וכמה שיפורים עשה Policy Iteration?
2. כל הערכה בנגן לוקחת עשרות סבבים. למה סבב של הערכה **זול** יותר מסבב של Value Iteration? (רמז: כמה פעולות בודקים בכל משבצת?)
3. למה שני האלגוריתמים מגיעים בדיוק לאותה מדיניות?
'''),
        ans('q3'),
        # ---------------- part 6 ----------------
        md('''
## חלק 6: ✨ בונוס: המכונית

ה-MDP של המכונית משיעור 6 הוא לא לוח, אלא מילון קטן של מעברים. `dict_value_iteration` (מוכנה) מריצה עליו Value Iteration. השלימו את המעברים.

| מצב | פעולה | תוצאה |
|---|---|---|
| קר | לאט | נשארים קרים, ‎+1 |
| קר | מהר | חצי קר, חצי חם, ‎+2 |
| חם | לאט | חצי קר, חצי חם, ‎+1 |
| חם | מהר | התחממות יתר, ‎−10 |
'''),
        code(DICT_VI),
        c(TASK6),
        code('''check_race(RACE)'''),
        guarded('''for gamma in [1.0, 0.9]:
    Vs = dict_value_iteration(RACE, gamma, 5)
    print(f'gamma = {gamma}:', ' | '.join(f"V{k}: cool {V['cool']:.2f}, warm {V['warm']:.2f}" for k, V in enumerate(Vs)))
print('gamma = 0.9, converged:', {s: round(v, 2) for s, v in dict_value_iteration(RACE, 0.9)[-1].items()})''',
                'race (bonus)'),
        md('''
### ❓ שאלה 4: ✨ בלי היוון

1. מה קורה ל-V<sub>k</sub> של המכונית בלי היוון (γ = 1), כש-k גדל? למה?
2. למה עם γ = 0.9 הערכים מתכנסים? מה זה אומר על הצורך ב-γ קטן מ-1 (או בסוף ידוע מראש)?
'''),
        ans('q4'),
        md('''
## סיכום והגשה

- **משוואות בלמן:** כל ערך הוא צעד אחד של Expectimax מעל ערכים של המצבים הבאים (`q_value`).
- **Value Iteration:** בונים את V₀, V₁, … עד שמתכנס. המדיניות מתייצבת הרבה לפני הערכים.
- **חילוץ מדיניות:** argmax של צעד אחד. מתוך Q זה הכי קל.
- **הערכת מדיניות:** אותו חישוב בלי max. מדויק, ולא ממוצע של ריצות.
- **Policy Iteration:** הערכה ושיפור לסירוגין, עד שהמדיניות לא משתנה. אותה תוצאה כמו Value Iteration.

הריצו את התא האחרון. לפני ההגשה ודאו שכל השורות מסומנות ✅ (הבונוס לא חובה), ושכל התשובות ✍️ מלאות.
'''),
        code('final_report()'),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {'colab': {'provenance': [], 'toc_visible': True},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                   'language_info': {'name': 'python'}}
    return nb


for sol, name in [(False, 'HM_07.ipynb'), (True, 'HM_07_solution.ipynb')]:
    nbf.write(build(sol), f'{OUT}/{name}')
print('written', OUT)
