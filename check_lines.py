import pandas as pd
import os

data_dir = '示例卷轴'
file_path = os.path.join(data_dir, 'WinLine.xlsx')
try:
    df = pd.read_excel(file_path)
    print(f"Total rows: {len(df)}")
    print(df)
except Exception as e:
    print(e)
