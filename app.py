import joblib
import streamlit as st

from featurize import PROCESSING, SYMBOLS, composition_key, describe, featurize, parse_formula

st.set_page_config(page_title="HEA Phase Predictor", page_icon=None, layout="wide")

COLORS = {"BCC": "#3373D9", "FCC": "#ED8026", "other": "#738091"}
EXAMPLES = ["CoCrFeMnNi", "AlCoCrFeNi", "CoCrFeNi", "HfNbTaTiZr", "MoNbTaW"]

st.markdown(
    """
<style>
.stApp { background: #F6F7F9; }
.block-container { max-width: 1344px; padding-top: 2.2rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 16px; }
.hea-title { display: flex; align-items: center; gap: 12px; font-size: 28px; font-weight: 700; color: #171A21; }
.hea-logo { width: 36px; height: 36px; border-radius: 8px; background: #3349C9; color: #fff; font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.hea-sub { color: #66707D; font-size: 15px; line-height: 22px; max-width: 880px; margin: 6px 0 22px 0; }
.hea-h { font-size: 18px; font-weight: 600; color: #171A21; margin-bottom: 4px; }
.hea-h2 { font-size: 15px; font-weight: 600; color: #171A21; margin: 22px 0 10px 0; }
.hea-verdict { display: flex; gap: 24px; align-items: center; }
.hea-badge { border-radius: 14px; padding: 18px 28px; text-align: center; color: #fff; min-width: 150px; }
.hea-badge small { display: block; font-size: 12px; font-weight: 500; opacity: 0.85; }
.hea-badge b { font-size: 44px; line-height: 1.15; }
.hea-vt h3 { margin: 0 0 6px 0; font-size: 20px; font-weight: 600; color: #171A21; }
.hea-vt p { margin: 0 0 8px 0; font-size: 14px; line-height: 21px; color: #66707D; }
.pill { display: inline-block; border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 500; }
.pill-ok { background: #E1F5E6; color: #1A6B38; }
.pill-new { background: #EEF0F8; color: #3349C9; }
.prow { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; font-size: 14px; color: #171A21; }
.prow .l { width: 50px; font-weight: 500; }
.prow .t { flex: 1; height: 12px; border-radius: 6px; background: #F0F2F8; overflow: hidden; }
.prow .b { height: 12px; border-radius: 6px; }
.prow .p { width: 48px; font-weight: 600; text-align: right; }
.tiles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
.tile { background: #F0F2F8; border-radius: 12px; padding: 14px 16px; }
.tile .k { font-size: 12px; font-weight: 500; color: #66707D; }
.tile .v { font-size: 24px; font-weight: 700; color: #171A21; }
.tile .u { font-size: 12px; font-weight: 400; color: #66707D; margin-left: 4px; }
.gauge { position: relative; height: 64px; margin-top: 6px; }
.gauge .zone { position: absolute; top: 24px; height: 12px; }
.gauge .tick { position: absolute; top: 20px; width: 2px; height: 20px; background: #171A21; }
.gauge .lab { position: absolute; top: 44px; font-size: 11px; color: #66707D; transform: translateX(-50%); white-space: nowrap; }
.gauge .dot { position: absolute; top: 21px; width: 12px; height: 12px; border-radius: 50%; background: #fff; border: 3px solid #171A21; transform: translateX(-50%); }
.gauge .me { position: absolute; top: 2px; font-size: 11px; font-weight: 600; color: #171A21; transform: translateX(-50%); white-space: nowrap; }
.note { border-radius: 10px; padding: 12px 16px; font-size: 13px; line-height: 19px; margin-top: 14px; }
.note-warn { background: #FFF5E0; color: #8C5C0D; }
.note-info { background: #F0F2F8; color: #3A4252; }
.facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 32px; }
.facts .k { font-size: 12px; font-weight: 500; color: #66707D; }
.facts .v { font-size: 14px; font-weight: 600; color: #171A21; line-height: 20px; }
.err { background: #FDECEC; color: #9B1C1C; border-radius: 10px; padding: 14px 16px; font-size: 14px; }
.sup { background: #F0F2F8; border-radius: 10px; padding: 12px 14px; font-size: 12px; line-height: 18px; color: #66707D; }
.sup b { color: #171A21; display: block; margin-bottom: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load():
    bundle = joblib.load("model.joblib")
    return bundle["model"], bundle["meta"]


model, meta = load()
classes = list(model.classes_)
VEC_LO, VEC_HI = 4.0, 10.0


def pct(v):
    return (min(max(v, VEC_LO), VEC_HI) - VEC_LO) / (VEC_HI - VEC_LO) * 100


def pick(example):
    st.session_state["formula"] = example
    st.session_state["submitted"] = (example, st.session_state["proc"])


def submit():
    st.session_state["submitted"] = (st.session_state["formula"], st.session_state["proc"])


st.session_state.setdefault("formula", "CoCrFeMnNi")
st.session_state.setdefault("proc", "CAST")
st.session_state.setdefault("submitted", ("CoCrFeMnNi", "CAST"))

st.markdown(
    '<div class="hea-title"><div class="hea-logo">HEA</div>High-Entropy Alloy Phase Predictor</div>'
    f'<div class="hea-sub">Enter a composition and a processing route to estimate whether the alloy forms a BCC, FCC or other phase. '
    f'Trained on {meta["n_alloys"]} unique compositions with composition-grouped validation.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2], gap="medium")

with left:
    with st.container(border=True):
        st.markdown('<div class="hea-h">Alloy</div>', unsafe_allow_html=True)
        st.text_input(
            "Composition",
            key="formula",
            help="Element symbols with optional amounts, for example Al0.5 Co1 Cr1 Fe1 Ni1",
            on_change=submit,
        )
        st.caption("Element symbols with optional amounts, e.g. Al0.5 Co1 Cr1 Fe1 Ni1")
        st.selectbox("Processing route", PROCESSING, key="proc", on_change=submit)
        st.markdown("<div style='font-size:14px;font-weight:500;margin:6px 0 4px'>Try an example</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            cols[i % 2].button(ex, key="ex_" + ex, on_click=pick, args=(ex,), use_container_width=True)
        st.button("Predict phase", type="primary", use_container_width=True, on_click=submit)
        st.markdown(
            f'<div class="sup"><b>Supported elements ({len(SYMBOLS)})</b>{" ".join(SYMBOLS)}</div>',
            unsafe_allow_html=True,
        )

with right:
    with st.container(border=True):
        formula, proc = st.session_state["submitted"]
        try:
            fractions = parse_formula(formula)
        except ValueError as e:
            st.markdown(f'<div class="err">{e}</div>', unsafe_allow_html=True)
            fractions = None

        if fractions is not None:
            X = featurize(fractions, proc)
            proba = model.predict_proba(X)[0]
            probs = dict(zip(classes, proba))
            best = max(probs, key=probs.get)
            d = describe(fractions)
            known = composition_key(formula) in set(meta["known_keys"])
            seen = (
                '<span class="pill pill-ok">This composition is in the training data, so the probabilities are optimistic</span>'
                if known
                else '<span class="pill pill-new">This composition is not in the training data</span>'
            )
            vec = d["mean_valence_electrons"]
            if best == "FCC":
                lead = "Mean VEC is in the range where FCC is favoured."
            elif best == "BCC":
                lead = "Mean VEC is in the range where BCC is favoured."
            else:
                lead = "Mean VEC sits between the BCC and FCC regions, or the alloy is complex enough that the model expects a secondary phase."
            st.markdown(
                f"""
<div class="hea-verdict">
  <div class="hea-badge" style="background:{COLORS[best]}"><small>Predicted phase</small><b>{best}</b></div>
  <div class="hea-vt"><h3>{formula.strip()}, {proc.lower()}</h3><p>{lead} Top class probability {probs[best]:.0%}.</p>{seen}</div>
</div>
<div class="hea-h2">Class probabilities</div>
"""
                + "".join(
                    f'<div class="prow"><span class="l">{c}</span><span class="t"><span class="b" style="display:block;width:{max(probs[c] * 100, 1.5):.1f}%;background:{COLORS[c]}"></span></span><span class="p">{probs[c]:.0%}</span></div>'
                    for c in ["BCC", "FCC", "other"]
                    if c in probs
                )
                + '<div class="hea-h2">Descriptors</div><div class="tiles">'
                + "".join(
                    f'<div class="tile"><div class="k">{k}</div><div class="v">{v}<span class="u">{u}</span></div></div>'
                    for k, v, u in [
                        ("Mean VEC", f"{vec:.2f}", ""),
                        ("Size mismatch", f"{d['delta']:.1f}", "%"),
                        ("Mixing entropy", f"{d['delta_S']:.1f}", "J/mol/K"),
                        ("Density", f"{d['calculated density']:.2f}", "g/cm3"),
                        ("Elements", f"{d['n_elements']}", ""),
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )

            mv = meta["class_mean_vec"]
            zone_b, zone_f = (mv["BCC"] + mv["other"]) / 2, (mv["other"] + mv["FCC"]) / 2
            zones = [
                (VEC_LO, zone_b, COLORS["BCC"]),
                (zone_b, zone_f, COLORS["other"]),
                (zone_f, VEC_HI, COLORS["FCC"]),
            ]
            gauge = "".join(
                f'<div class="zone" style="left:{pct(a):.2f}%;width:{pct(b) - pct(a):.2f}%;background:{c}"></div>' for a, b, c in zones
            )
            for name, label in [("BCC", "BCC mean"), ("other", "other mean"), ("FCC", "FCC mean")]:
                x = pct(mv[name])
                gauge += f'<div class="tick" style="left:{x:.2f}%"></div><div class="lab" style="left:{x:.2f}%">{label} {mv[name]:.1f}</div>'
            gauge += f'<div class="dot" style="left:{pct(vec):.2f}%"></div><div class="me" style="left:{pct(vec):.2f}%">this alloy {vec:.1f}</div>'
            st.markdown(
                f'<div class="hea-h2">Where this alloy sits on the VEC scale</div><div class="gauge">{gauge}</div>',
                unsafe_allow_html=True,
            )

            notes = []
            lo, hi = meta["n_elements_range"]
            if not lo <= d["n_elements"] <= hi:
                notes.append(f"Training alloys have {lo} to {hi} elements, so this composition is outside the range the model has seen.")
            rare = [s for s in fractions if meta["element_alloys"].get(s, 0) < 5]
            if rare:
                notes.append(
                    "Few training alloys contain " + ", ".join(rare) + ", so the prediction for this composition is less reliable."
                )
            notes.append(
                "Predictions are statistical estimates from composition and a coarse processing label. Heat treatment and secondary phases are not modelled, and FCC recall in validation is only about 0.6."
            )
            st.markdown(
                "".join(f'<div class="note note-warn">{n}</div>' for n in notes),
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(
        f"""
<div class="facts">
  <div><div class="k">Model</div><div class="v">Random forest, class-balanced, 42 features</div></div>
  <div><div class="k">Validation</div><div class="v">5-fold grouped by composition, 5 seeds</div></div>
  <div><div class="k">Alloy macro F1</div><div class="v">0.754 +/- 0.012 (baseline 0.254)</div></div>
  <div><div class="k">Data</div><div class="v">{meta["n_rows"]} records, {meta["n_alloys"]} unique compositions</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
