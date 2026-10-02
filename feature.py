import pandas as pd 
import re
from pymatgen.core import Element
import numpy as np

R= 8.3145

data = pd.read_csv("cleaned_data.csv").reset_index(drop=True)
pattern = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?'

comp_list = []
for formula in data['formula']:
    matches = re.findall(pattern, formula)
    comp_list.append({m[0]: float(m[1]) if m[1] else 1.0 for m in matches})

comp_df = pd.DataFrame(comp_list, index=data.index).fillna(0)
comp_df = comp_df.div(comp_df.sum(axis=1), axis=0)

df = pd.concat([data, comp_df], axis=1)

Properties = pd.DataFrame(columns=['atomic_radius', 'electronegativity', 'melting_point'])  
for element in comp_df.columns: 
    el = Element(element)
    Properties.loc[element, 'atomic_radius'] = el.atomic_radius
    Properties.loc[element, 'electronegativity'] = el.X 
    Properties.loc[element, 'melting_point'] = el.melting_point
Properties = Properties.astype(float)

vec = {
    'Al': 3, 'B': 3, 'C': 4, 'Co': 9, 'Cr': 6, 'Cu': 11, 'Fe': 8,
    'Ga': 3, 'Hf': 4, 'Li': 1, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'Nb': 5,
    'Nd': 3, 'Ni': 10, 'Pd': 10, 'Re': 7, 'Sc': 3, 'Si': 4, 'Sn': 4,
    'Ta': 5, 'Ti': 4, 'V': 5, 'W': 6, 'Y': 3, 'Zn': 12, 'Zr': 4,
}

Properties['valence_electrons'] = Properties.index.map(vec)

mean_props = comp_df.dot(Properties)
mean_props.columns = ['mean_' + c for c in Properties.columns]

r= Properties['atomic_radius']
chi = Properties['electronegativity']

mean_r = mean_props['mean_atomic_radius']   
mean_chi = mean_props['mean_electronegativity']

c = comp_df.values
r_arr = r.values
rbar = mean_r.values[:, None]

rel =1-(r_arr/ rbar)
weighted = c * rel**2
total = weighted.sum(axis=1)
delta= np.sqrt(total)*100

e = chi.values
chi_bar = mean_chi.values[:, None]
rel_chi = e-chi_bar
weighted_chi = c * rel_chi**2
total_chi = weighted_chi.sum(axis=1)
delta_chi = np.sqrt(total_chi)

c_safe = comp_df.replace(0,1)
delta_S = -R * (comp_df * np.log(c_safe)).sum(axis=1)

n_elements = (comp_df>0).sum(axis=1)

delta = pd.Series(delta, index=data.index, name='delta')
delta_chi = pd.Series(delta_chi, index=data.index, name='delta_chi')
delta_S = pd.Series(delta_S, index=data.index, name='delta_S')
n_elements = pd.Series(n_elements, index=data.index, name='n_elements')

proc = pd.get_dummies(data['processing method'], prefix='proc', dtype=int)
feature = pd.concat([comp_df, mean_props, delta, delta_chi, delta_S, n_elements, proc, data[['calculated density']]], axis=1)

feature.to_csv('features.csv')
data['bcc/fcc/other'].to_csv('target.csv')