import pandas as pd
raw = pd.read_csv("High Entropy Alloy Properties.csv")  

import re

def clean_col(col):
    if col == 'FORMULA':
        return col
    name = col.split(':')[1].strip()         
    name = re.sub(r'\s*\(.*?\)', '', name)     
    return name.strip()

raw.columns = [clean_col(col) for col in raw.columns]
raw.columns = raw.columns.str.lower()

raw = raw.drop(columns=['reference id', 'exp. density', 'exp. young modulus', 'type of test', 
                        'test temperature', 'o content', 'n content','c content', 'elongation plastic', 
                        'grain size', 'uts','calculated young modulus', 'elongation'])

raw = raw.dropna(subset=['processing method', 'microstructure'])

clean = raw[['formula','processing method', 'calculated density', 'bcc/fcc/other', 'microstructure', 'hv','ys']]
clean.to_csv('cleaned_data.csv', index=False)