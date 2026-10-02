import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

R = 8.3145

ELEMENTS = {
    "Al": (1.25, 1.61, 933.47, 3, 26.9815386, 2.7),
    "Co": (1.35, 1.88, 1768.0, 9, 58.933195, 8.9),
    "Fe": (1.4, 1.83, 1811.0, 8, 55.845, 7.874),
    "Ni": (1.35, 1.91, 1728.0, 10, 58.6934, 8.908),
    "Si": (1.1, 1.9, 1687.0, 4, 28.0855, 2.33),
    "Cr": (1.4, 1.66, 2180.0, 6, 51.9961, 7.14),
    "Mn": (1.4, 1.55, 1519.0, 7, 54.938045, 7.47),
    "Nb": (1.45, 1.6, 2750.0, 5, 92.90638, 8.57),
    "Mo": (1.45, 2.16, 2896.0, 6, 95.94, 10.28),
    "Ti": (1.4, 1.54, 1941.0, 4, 47.867, 4.507),
    "Cu": (1.35, 1.9, 1357.77, 11, 63.546, 8.92),
    "C": (0.7, 2.55, 3800.0, 4, 12.0107, 2.267),
    "V": (1.35, 1.63, 2183.0, 5, 50.9415, 6.11),
    "Zr": (1.55, 1.33, 2128.0, 4, 91.224, 6.511),
    "Nd": (1.85, 1.14, 1297.0, 3, 144.242, 6.8),
    "Y": (1.8, 1.22, 1799.0, 3, 88.90585, 4.472),
    "Sn": (1.45, 1.96, 505.08, 4, 118.71, 7.31),
    "Li": (1.45, 0.98, 453.69, 1, 6.941, 0.535),
    "Mg": (1.5, 1.31, 923.0, 2, 24.305, 1.738),
    "Zn": (1.35, 1.65, 692.68, 12, 65.409, 7.14),
    "Ta": (1.45, 1.5, 3290.0, 5, 180.94788, 16.65),
    "Hf": (1.55, 1.3, 2506.0, 4, 178.49, 13.31),
    "W": (1.35, 2.36, 3695.0, 6, 183.84, 19.25),
    "Re": (1.35, 1.9, 3459.0, 7, 186.207, 21.02),
    "Pd": (1.4, 2.2, 1828.05, 10, 106.42, 12.023),
    "B": (0.85, 2.04, 2349.0, 3, 10.811, 2.46),
    "Sc": (1.6, 1.36, 1814.0, 3, 44.955912, 2.985),
    "Ga": (1.3, 1.81, 302.91, 3, 69.723, 5.904),
}

SYMBOLS = list(ELEMENTS)
PROCESSING = ["ANNEAL", "CAST", "OTHER", "POWDER", "WROUGHT"]
FEATURE_COLUMNS = (
    SYMBOLS
    + [
        "mean_atomic_radius",
        "mean_electronegativity",
        "mean_melting_point",
        "mean_valence_electrons",
        "delta",
        "delta_chi",
        "delta_S",
        "delta_H",
        "n_elements",
    ]
    + ["proc_" + p for p in PROCESSING]
    + ["calculated density"]
)
PHYSICS = ["delta", "delta_H", "delta_S", "mean_valence_electrons", "mean_melting_point", "delta_chi"]
MODEL_COLUMNS = PHYSICS + ["proc_" + p for p in PROCESSING]

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")
_PROPS = np.array([ELEMENTS[s] for s in SYMBOLS], dtype=float)
_RADIUS, _CHI, _MELT, _VEC, _MASS, _DENSITY = _PROPS.T


@lru_cache(maxsize=1)
def _miedema():
    table = pd.read_csv(Path(__file__).with_name("miedema_matrix.csv"), index_col=0)
    return table.loc[SYMBOLS, SYMBOLS].to_numpy(dtype=float)


def parse_formula(text):
    cleaned = re.sub(r"\s+", "", text or "")
    if not cleaned:
        raise ValueError("Enter a composition such as AlCoCrFeNi or Al0.5 Co1 Cr1 Fe1 Ni1.")
    tokens = _TOKEN.findall(cleaned)
    if "".join(sym + num for sym, num in tokens) != cleaned:
        raise ValueError(
            "Could not read that composition. Use element symbols with optional amounts, "
            "for example CoCrFeMnNi or Al0.5 Co1 Cr1 Fe1 Ni1. Parentheses are not supported."
        )
    amounts = {}
    for sym, num in tokens:
        value = float(num) if num else 1.0
        if value <= 0:
            raise ValueError(f"The amount of {sym} must be greater than zero.")
        amounts[sym] = amounts.get(sym, 0.0) + value
    unknown = [s for s in amounts if s not in ELEMENTS]
    if unknown:
        raise ValueError(
            "Unsupported element(s): " + ", ".join(unknown) + ". Supported elements: " + ", ".join(SYMBOLS) + "."
        )
    if len(amounts) < 2:
        raise ValueError("Enter at least two elements.")
    total = sum(amounts.values())
    return {s: v / total for s, v in amounts.items()}


def composition_key(text):
    amounts = {}
    for sym, num in _TOKEN.findall(text):
        amounts[sym] = amounts.get(sym, 0.0) + (float(num) if num else 1.0)
    total = sum(amounts.values())
    return "_".join(f"{s}{amounts[s] / total:.3f}" for s in sorted(amounts))


def describe(fractions):
    c = np.array([fractions.get(s, 0.0) for s in SYMBOLS])
    rbar = c @ _RADIUS
    chibar = c @ _CHI
    mass = c @ _MASS
    nz = c > 0
    return {
        "c": c,
        "mean_atomic_radius": rbar,
        "mean_electronegativity": chibar,
        "mean_melting_point": c @ _MELT,
        "mean_valence_electrons": c @ _VEC,
        "delta": 100 * np.sqrt(np.sum(c * (1 - _RADIUS / rbar) ** 2)),
        "delta_chi": np.sqrt(np.sum(c * (_CHI - chibar) ** 2)),
        "delta_S": -R * np.sum(c[nz] * np.log(c[nz])),
        "delta_H": float(2 * c @ _miedema() @ c),
        "n_elements": int(nz.sum()),
        "calculated density": mass / np.sum(c * _MASS / _DENSITY),
    }


def featurize(fractions, processing):
    if processing not in PROCESSING:
        raise ValueError("Unknown processing route: " + str(processing))
    d = describe(fractions)
    row = {s: v for s, v in zip(SYMBOLS, d["c"])}
    for key in [
        "mean_atomic_radius",
        "mean_electronegativity",
        "mean_melting_point",
        "mean_valence_electrons",
        "delta",
        "delta_chi",
        "delta_S",
        "delta_H",
        "n_elements",
        "calculated density",
    ]:
        row[key] = d[key]
    for p in PROCESSING:
        row["proc_" + p] = int(p == processing)
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def featurize_table(formulas, processing):
    frames = [featurize(parse_formula(f), p) for f, p in zip(formulas, processing)]
    return pd.concat(frames, ignore_index=True)
