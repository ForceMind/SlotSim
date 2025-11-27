import pandas as pd
import os

data_dir = 'test'
reel_path = os.path.join(data_dir, 'SlotNormal.xlsx')

try:
    df = pd.read_excel(reel_path, header=1)
    
    for i in range(1, 6):
        col_sym = f'Reels{i}'
        col_w = f'&Weight{i}'
        
        if col_sym not in df.columns:
            continue
            
        # Filter valid rows
        reel_data = df[[col_sym, col_w]].dropna()
        
        # Check for ID 11
        wild_rows = reel_data[reel_data[col_sym] == 11]
        
        total_weight = reel_data[col_w].sum()
        wild_weight = wild_rows[col_w].sum()
        
        print(f"Reel {i}: Total Weight = {total_weight}")
        if not wild_rows.empty:
            print(f"  Wild (ID 11) found at indices: {wild_rows.index.tolist()}")
            print(f"  Wild Weight = {wild_weight} ({wild_weight/total_weight*100:.2f}%)")
        else:
            print("  Wild (ID 11) NOT FOUND!")
            
except Exception as e:
    print(e)
