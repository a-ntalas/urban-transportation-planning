import pandas as pd
import numpy as np

# load Excel file 
df = pd.read_excel("xls/od.xlsx", index_col=0)

# set diagonal = 0
np.fill_diagonal(df.values, 0)

print("OD matrix with diagonal = 0:")
print(df)

pd.DataFrame(df).to_csv("xls/od.csv", index=False)