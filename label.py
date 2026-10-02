import pandas as pd

data = pd.read_csv('cleaned_data.csv')
micro = data['microstructure'].str.get_dummies(sep='+')
micro = micro.rename(columns={'Sec.':'Sec'})
target_cols = ['HCP', 'Other', 'L12']
micro['rare'] = (micro[target_cols] == 1).any(axis=1).astype(int)
micro = micro.drop(columns=target_cols)

micro.to_csv('labels.csv', index=True)