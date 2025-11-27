import pandas as pd
import os

data_dir = 'test'
reel_path = os.path.join(data_dir, 'SlotNormal.xlsx')

try:
    df = pd.read_excel(reel_path, header=1)
    
    # Analyze Reel 1
    col_sym = 'Reels1'
    col_w = '&Weight1'
    
    reel_data = df[[col_sym, col_w]].dropna()
    total_weight = reel_data[col_w].sum()
    
    print(f"Reel 1 Total Weight: {total_weight}")
    
    sym_weights = reel_data.groupby(col_sym)[col_w].sum()
    
    print("\nSymbol Probabilities (Reel 1):")
    for sym, w in sym_weights.items():
        prob = w / total_weight
        print(f"ID {int(sym)}: {prob*100:.2f}%")
        
except Exception as e:
    print(e)
