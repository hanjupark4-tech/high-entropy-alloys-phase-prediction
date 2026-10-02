import pandas as pd

from featurize import SYMBOLS

pairs = pd.read_csv("MiedemaLiquidDeltaHf.tsv", sep=r"\s+")
pairs = pairs[pairs[["elem_A", "elem_B"]].isin(SYMBOLS).all(axis=1)]
tri = pairs.pivot(index="elem_B", columns="elem_A", values="delta_Hf").reindex(index=SYMBOLS, columns=SYMBOLS).fillna(0)
matrix = tri + tri.T
matrix.to_csv("miedema_matrix.csv")
print(matrix.shape, "pairs with data:", int((matrix.values != 0).sum() // 2))
