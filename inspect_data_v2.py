import pandas as pd
import os

data_dir = 'test'
files = ['Payout.xlsx', 'SlotNormal.xlsx']

for file in files:
    file_path = os.path.join(data_dir, file)
    print(f"--- {file} ---")
    try:
        df = pd.read_excel(file_path, header=1)
        print(df.head(20))
        print("\n")
    except Exception as e:
        print(f"Error reading {file}: {e}")
