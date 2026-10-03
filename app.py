import joblib
import streamlit as st

from featurize import MODEL_COLUMNS, SYMBOLS, composition_key, describe, featurize, parse_formula

st.set_page_config(page_title="HEA Phase Predictor", page_icon=None, layout="wide")

ACCENT = "#1B3A6B"
MID = "#9FB1CC"
ORDER = ["ANNEAL", "CAST", "WROUGHT", "POWDER", "OTHER"]
EXAMPLES = ["CoCrFeMnNi", "AlCoCrFeNi", "Al0.5CoCrFeNi", "HfNbTaTiZr", "MoNbTaW"]
GITHUB = "https://github.com/hanjupark4-tech/high-entropy-alloys-phase-prediction"
COMBO_NAMES = {"rare_combo": ["Other combinations"], "none": ["None of the five"]}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
.stApp { background: #F9FAFB; }
.block-container, div[data-testid="stMainBlockContainer"] { max-width: 1440px; padding: 0 40px 32px 40px !important; }
div[data-testid="stAppViewContainer"] > .main, section.main { padding-top: 0; }
div[data-baseweb="input"] { background: #fff; border: 1px solid rgba(15,23,42,0.12); }
div[data-baseweb="input"] input, div[data-baseweb="base-input"] { background: #fff !important; }
header[data-testid="stHeader"] { display: none; }
footer { visibility: hidden; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border: 1px solid rgba(15,23,42,0.08); border-radius: 12px; }
.topbar { display: flex; justify-content: space-between; align-items: center; height: 64px; margin: 0 -40px 32px -40px; padding: 0 40px; background: #fff; border-bottom: 1px solid rgba(15,23,42,0.08); }
.brand { display: flex; align-items: center; gap: 12px; font-size: 16px; font-weight: 600; color: #111827; }
.mark { width: 28px; height: 28px; border-radius: 12px; background: #1B3A6B; color: #fff; font-size: 14px; font-weight: 600; display: flex; align-items: center; justify-content: center; }
.links { display: flex; gap: 28px; font-size: 14px; color: #6B7280; }
.links a { color: #6B7280; text-decoration: none; }
.h-card { font-size: 16px; font-weight: 600; color: #111827; margin-bottom: 8px; }
.h1 { font-size: 28px; line-height: 34px; font-weight: 600; color: #111827; }
.sub { font-size: 14px; color: #6B7280; margin-top: 4px; }
.head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
.sel { background: #F3F4F6; border-radius: 12px; padding: 8px 14px; font-size: 13px; font-weight: 500; color: #6B7280; white-space: nowrap; }
.card { background: #fff; border: 1px solid rgba(15,23,42,0.08); border-radius: 12px; padding: 24px; margin-bottom: 24px; }
.card-h { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.card-h b { font-size: 16px; font-weight: 600; color: #111827; }
.card-h span { font-size: 13px; color: #9CA3AF; }
.prow { display: flex; align-items: center; gap: 16px; padding: 8px 0; font-size: 14px; color: #111827; }
.prow .l { width: 56px; font-weight: 500; }
.prow .t { flex: 1; height: 8px; border-radius: 12px; background: #F3F4F6; overflow: hidden; }
.prow .b { display: block; height: 8px; border-radius: 12px; }
.prow .p { width: 48px; font-weight: 500; text-align: right; }
.crow { display: flex; align-items: center; gap: 16px; height: 44px; font-size: 14px; }
.crow .r { width: 16px; font-weight: 500; color: #9CA3AF; }
.crow .tags { flex: 1; display: flex; gap: 6px; }
.crow .t { width: 240px; height: 8px; border-radius: 12px; background: #F3F4F6; overflow: hidden; }
.crow .b { display: block; height: 8px; border-radius: 12px; }
.crow .p { width: 48px; font-weight: 500; text-align: right; color: #111827; }
.tag { background: #EEF2F8; color: #1B3A6B; border-radius: 12px; padding: 4px 12px; font-size: 13px; font-weight: 500; }
.drow { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(15,23,42,0.08); font-size: 14px; }
.drow span { color: #6B7280; }
.drow b { font-weight: 500; color: #111827; }
.note { border-radius: 12px; padding: 12px 16px; font-size: 13px; line-height: 19px; margin-bottom: 8px; background: #F3F4F6; color: #374151; }
.err { background: #FEF2F2; color: #991B1B; border-radius: 12px; padding: 14px 16px; font-size: 14px; }
.sup { background: #F3F4F6; border-radius: 12px; padding: 12px 14px; font-size: 12px; line-height: 18px; color: #6B7280; margin-top: 12px; }
.sup b { color: #111827; display: block; margin-bottom: 4px; }
.facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 32px; }
.facts .k { font-size: 12px; font-weight: 500; color: #6B7280; }
.facts .v { font-size: 14px; font-weight: 500; color: #111827; line-height: 20px; }
div[data-testid="stButton"] button[kind="primary"] { background: #1B3A6B; border: none; border-radius: 12px; }
div[data-testid="stButton"] button { border-radius: 12px; }
div[data-baseweb="input"], div[data-baseweb="input"] > div { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load():
    bundle = joblib.load("model.joblib")
    return bundle["phase_models"], bundle["combo_model"], bundle["meta"]


phase_models, combo_model, meta = load()
phases = meta["phases"]
cv = meta["cv"]


def pick():
    example = st.session_state["example"]
    if example:
        st.session_state["formula"] = example
        st.session_state["submitted"] = (example, st.session_state["proc"] or st.session_state["submitted"][1])
        st.session_state["example"] = None


def submit():
    st.session_state["submitted"] = (st.session_state["formula"], st.session_state["proc"] or st.session_state["submitted"][1])


st.session_state.setdefault("formula", "Al0.5 Co1 Cr1 Fe1 Ni1")
st.session_state.setdefault("proc", "ANNEAL")
st.session_state.setdefault("submitted", ("Al0.5 Co1 Cr1 Fe1 Ni1", "ANNEAL"))

st.markdown(
    f'<div class="topbar"><div class="brand"><div class="mark">H</div>HEA Phase Predictor</div>'
    f'<div class="links"><span>Method</span><span>Data</span><a href="{GITHUB}" target="_blank">GitHub</a></div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([357, 979], gap="medium")

formula, proc = st.session_state["submitted"]
try:
    fractions = parse_formula(formula)
    error = None
except ValueError as e:
    fractions = None
    error = str(e)

d = describe(fractions) if fractions is not None else None

with left:
    with st.container(border=True):
        st.markdown('<div class="h-card">Alloy</div>', unsafe_allow_html=True)
        st.text_input(
            "Composition",
            key="formula",
            help="Element symbols with optional amounts, for example Al0.5 Co1 Cr1 Fe1 Ni1",
            on_change=submit,
        )
        st.pills("Processing route", ORDER, key="proc", selection_mode="single", format_func=str.title, on_change=submit)
        st.button("Predict phases", type="primary", use_container_width=True, on_click=submit)
        st.pills("Try an example", EXAMPLES, key="example", selection_mode="single", on_change=pick)
        with st.expander(f"Supported elements ({len(SYMBOLS)})"):
            st.markdown(f'<div style="font-size:12px;line-height:18px;color:#6B7280">{" ".join(SYMBOLS)}</div>', unsafe_allow_html=True)

    if d is not None:
        minus = lambda x, spec: format(x, spec).replace("-", "&minus;")
        rows = [
            ("Atomic size mismatch, &delta;", f"{d['delta']:.2f} %"),
            ("Mixing enthalpy, &Delta;H<sub>mix</sub>", minus(d["delta_H"], ".2f") + " kJ/mol"),
            ("Mixing entropy, &Delta;S", f"{d['delta_S']:.2f} J/mol&middot;K"),
            ("Valence electron conc., VEC", f"{d['mean_valence_electrons']:.2f}"),
            ("Mean melting point, T<sub>m</sub>", f"{d['mean_melting_point']:,.0f} K"),
            ("Electronegativity diff., &Delta;&chi;", f"{d['delta_chi']:.3f}"),
            ("Strongest pair enthalpy, &Delta;H<sub>pair,min</sub>", minus(d["h_min_pair"], ".0f") + " kJ/mol"),
        ]
        st.markdown(
            '<div class="card"><div class="card-h"><b>Computed descriptors</b></div>'
            + "".join(f'<div class="drow"><span>{k}</span><b>{v}</b></div>' for k, v in rows)
            + "</div>",
            unsafe_allow_html=True,
        )

with right:
    if error is not None:
        st.markdown(f'<div class="err">{error}</div>', unsafe_allow_html=True)
    else:
        X = featurize(fractions, proc)[MODEL_COLUMNS]
        probs = {p: float(phase_models[p].predict_proba(X)[0, 1]) for p in phases}
        top_phase = max(probs, key=probs.get)
        combo_p = dict(zip(combo_model.classes_, combo_model.predict_proba(X)[0]))
        top3 = sorted(combo_p.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_p = top3[0][1]

        st.markdown(
            f'<div class="head"><div><div class="h1">Predicted phases</div>'
            f'<div class="sub">Probability that each phase is present in the microstructure</div></div>'
            f'<div class="sel">{formula.strip()} &nbsp;&middot;&nbsp; {proc.title()}</div></div>',
            unsafe_allow_html=True,
        )

        phase_rows = "".join(
            f'<div class="prow"><span class="l">{p}</span><span class="t"><span class="b" '
            f'style="width:{max(probs[p] * 100, 1.5):.1f}%;background:{ACCENT if p == top_phase else MID}"></span></span>'
            f'<span class="p">{probs[p]:.0%}</span></div>'
            for p in phases
        )
        st.markdown(
            f'<div class="card"><div class="card-h"><b>Phase probability</b><span>One model per phase</span></div>{phase_rows}</div>',
            unsafe_allow_html=True,
        )

        combo_rows = ""
        for i, (name, v) in enumerate(top3):
            labels = COMBO_NAMES.get(name, name.split("+"))
            tags = "".join(f'<span class="tag">{t}</span>' for t in labels)
            width = v / top_p * 100 if top_p > 0 else 0
            combo_rows += (
                f'<div class="crow"><span class="r">{i + 1}</span><span class="tags">{tags}</span>'
                f'<span class="t"><span class="b" style="width:{max(width, 1.5):.1f}%;background:{ACCENT if i == 0 else MID}"></span></span>'
                f'<span class="p">{v:.0%}</span></div>'
            )
        st.markdown(
            f'<div class="card"><div class="card-h"><b>Most likely combinations</b><span>Top 3 of {len(combo_model.classes_)}</span></div>{combo_rows}</div>',
            unsafe_allow_html=True,
        )

        notes = []
        if composition_key(formula) in set(meta["known_keys"]):
            notes.append("This composition is in the training data, so the probabilities are optimistic.")
        else:
            notes.append("This composition is not in the training data.")
        lo, hi = meta["n_elements_range"]
        if not lo <= d["n_elements"] <= hi:
            notes.append(f"Training alloys have {lo} to {hi} elements, so this composition is outside the range the model has seen.")
        rare = [s for s in fractions if meta["element_alloys"].get(s, 0) < 5]
        if rare:
            notes.append("Few training alloys contain " + ", ".join(rare) + ", so the prediction for this composition is less reliable.")
        notes.append(
            f"Laves and B2 are the hardest phases to predict: cross-validated average precision is {cv['ap']['Laves']:.2f} against "
            f"{cv['prevalence']['Laves']:.0%} prevalence for Laves and {cv['ap']['B2']:.2f} against {cv['prevalence']['B2']:.0%} for B2, "
            "so treat those bars as indicative."
        )
        st.markdown("".join(f'<div class="note">{n}</div>' for n in notes), unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(
        f"""<div class="facts">
<div><div class="k">Model</div><div class="v">Extra trees per phase and for phase combinations, seven physics descriptors plus five processing flags (12 inputs)</div></div>
<div><div class="k">Validation</div><div class="v">{cv["folds"]}</div></div>
<div><div class="k">Combination accuracy</div><div class="v">Top-1 {cv["top1"]:.2f}, top-3 {cv["top3"]:.2f} (majority baseline {cv["baseline"]:.2f})</div></div>
<div><div class="k">Data</div><div class="v">{meta["n_rows"]} records, {meta["n_alloys"]} unique compositions</div></div>
</div>""",
        unsafe_allow_html=True,
    )
