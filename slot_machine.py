import pandas as pd
import random
import numpy as np
from collections import defaultdict

class SlotMachine:
    def __init__(self, payout_path, reel_path, line_path, wild_id=None, free_spin_reel_path=None):
        self.payout_table = {}
        self.reels = []
        self.reel_weights = []
        self.total_weights = []
        self.free_reels = []
        self.free_reel_weights = []
        self.free_total_weights = []
        self.win_lines = []
        self.num_reels = 5
        self.num_rows = 3
        self.num_win_lines = 0
        self.wild_id = wild_id
        
        self._load_data(payout_path, reel_path, line_path, free_spin_reel_path)

    def _load_data(self, payout_path, reel_path, line_path, free_spin_reel_path=None):
        # 3. Load WinLines first to get count
        df_lines = pd.read_excel(line_path, header=1)
        # Columns: Id, Line
        for _, row in df_lines.iterrows():
            line_str = str(row['Line'])
            # Parse "1,1,1,1,1" -> [1, 1, 1, 1, 1]
            try:
                line_indices = [int(x.strip()) for x in line_str.split(',')]
                if len(line_indices) == self.num_reels:
                    self.win_lines.append(line_indices)
            except:
                continue
        
        self.num_win_lines = len(self.win_lines)
        if self.num_win_lines == 0:
            raise ValueError("No valid win lines found in WinLine.xlsx")

        # 1. Load Payout
        # Skip first row (header description) and use second row as header
        df_payout = pd.read_excel(payout_path, header=1)
        # Columns: Id, Payout2, Payout3, Payout4, Payout5
        for _, row in df_payout.iterrows():
            try:
                symbol_id = int(row['Id'])
                self.payout_table[symbol_id] = {}
                for count in range(2, 6):
                    col_name = f'Payout{count}'
                    if col_name in df_payout.columns:
                        val = row[col_name]
                        # User rule: value in table is for ALL lines, so single line is value / num_win_lines
                        self.payout_table[symbol_id][count] = float(val) / float(self.num_win_lines)
            except ValueError:
                continue

        # 2. Load Reels (Normal)
        self._load_reels(reel_path, self.reels, self.reel_weights, self.total_weights)
        
        # 4. Load Free Spin Reels (Optional)
        if free_spin_reel_path:
            self._load_reels(free_spin_reel_path, self.free_reels, self.free_reel_weights, self.free_total_weights)

    def _load_reels(self, path, reels_list, weights_list, total_weights_list):
        df_reels = pd.read_excel(path, header=1)
        # Columns: Reels1, &Weight1, Reels2, &Weight2, ...
        for i in range(1, self.num_reels + 1):
            reel_col = f'Reels{i}'
            weight_col = f'&Weight{i}'
            
            if reel_col not in df_reels.columns:
                break
                
            # Filter out NaNs (in case reels have different lengths)
            reel_data = df_reels[[reel_col, weight_col]].dropna()
            
            symbols = reel_data[reel_col].astype(int).tolist()
            weights = reel_data[weight_col].astype(int).tolist()
            
            reels_list.append(symbols)
            weights_list.append(weights)
            total_weights_list.append(sum(weights))

    def spin(self, use_free_reels=False):
        # Generate stop positions for each reel
        stops = []
        screen = np.zeros((self.num_rows, self.num_reels), dtype=int)
        
        current_reels = self.free_reels if use_free_reels and self.free_reels else self.reels
        current_weights = self.free_reel_weights if use_free_reels and self.free_reels else self.reel_weights
        current_totals = self.free_total_weights if use_free_reels and self.free_reels else self.total_weights
        
        for i in range(self.num_reels):
            # Weighted random selection
            r = random.uniform(0, current_totals[i])
            current_w = 0
            stop_idx = 0
            for idx, w in enumerate(current_weights[i]):
                current_w += w
                if r < current_w:
                    stop_idx = idx
                    break
            
            stops.append(stop_idx)
            
            # Fill screen column
            reel_len = len(current_reels[i])
            for row in range(self.num_rows):
                # Wrap around
                symbol_idx = (stop_idx + row) % reel_len
                screen[row, i] = current_reels[i][symbol_idx]
                
        return screen, stops

    def check_win(self, screen):
        total_win = 0.0
        win_details = [] # List of {line_id, symbol, count, win}
        
        for line_idx, line_indices in enumerate(self.win_lines):
            # Get symbols on this line
            line_symbols = []
            for col, row in enumerate(line_indices):
                if 0 <= row < self.num_rows:
                    line_symbols.append(screen[row, col])
                else:
                    line_symbols.append(-1) 
            
            if not line_symbols:
                continue
                
            # Determine the symbol to match (first non-wild)
            match_symbol = -1
            if self.wild_id is not None:
                for s in line_symbols:
                    if s != self.wild_id:
                        match_symbol = s
                        break
                # If all are wilds, match wild
                if match_symbol == -1:
                    match_symbol = self.wild_id
            else:
                match_symbol = line_symbols[0]
            
            # Count matches
            count = 0
            for s in line_symbols:
                if s == match_symbol or (self.wild_id is not None and s == self.wild_id):
                    count += 1
                else:
                    break
            
            # Lookup payout
            # If match_symbol is Wild (all wilds), use Wild payout
            # If match_symbol is regular, use regular payout
            # Note: Some games pay Wilds differently if they form a line of pure Wilds.
            # Here we assume if we matched 'match_symbol', we use its payout.
            # Exception: If we have 5 Wilds, match_symbol is Wild, so we use Wild payout.
            # If we have W W W A A, match_symbol is A. Count is 3. Payout for A(3).
            
            if match_symbol in self.payout_table:
                payouts = self.payout_table[match_symbol]
                if count in payouts and payouts[count] > 0:
                    # Payout value is for 20 lines. Single line win = Value / 20.
                    # This is a multiplier of Total Bet if we assume standard logic.
                    # If Total Bet = 100. Win = (Value / 20) * 100.
                    # But here we return the multiplier relative to Total Bet = 1.
                    # The caller will scale it.
                    
                    # Wait, previous logic: self.payout_table values are ALREADY divided by num_win_lines.
                    # So self.payout_table[sym][count] is the Win Amount for Bet=1.
                    
                    win_amount = payouts[count]
                    total_win += win_amount
                    win_details.append({
                        'line_index': line_idx,
                        'symbol': match_symbol,
                        'count': count,
                        'win': win_amount
                    })
                    
        return total_win, win_details

    def run_simulation(self, num_spins=100000, total_bet=100.0, 
                       scatter_id=None, fs_trigger_count=3, fs_award_count=10,
                       pity_streak_threshold=0):
        # total_bet is the amount bet per spin.
        # The payout table values (self.payout_table) are currently normalized to Bet=1.
        # So we need to multiply win by total_bet.
        
        total_win_amount = 0.0
        
        # Stats
        # symbol -> {3: count, 4: count, 5: count}
        symbol_hit_counts = defaultdict(lambda: defaultdict(int)) 
        symbol_win_amounts = defaultdict(float)
        win_dist = defaultdict(int)
        
        hits_count = 0
        max_win = 0.0
        
        win_amounts = []
        balance_history = [] # Track balance over time (assuming starting balance 0)
        current_balance = 0.0
        
        # Strategy State
        current_losing_streak = 0
        pity_activations_count = 0
        fs_triggers_count = 0
        
        print_interval = num_spins // 10
        if print_interval == 0: print_interval = 1
        
        for i in range(num_spins):
            # Pity System Intervention
            force_win = False
            if pity_streak_threshold > 0 and current_losing_streak >= pity_streak_threshold:
                force_win = True
                pity_activations_count += 1
            
            # Spin Loop (usually once, unless forced win)
            while True:
                screen, _ = self.spin(use_free_reels=False)
                win_multiplier, details = self.check_win(screen)
                
                # Check Free Spins Trigger
                fs_total_win = 0.0
                fs_triggered = False
                if scatter_id is not None:
                    # Count scatters
                    scatter_count = np.count_nonzero(screen == scatter_id)
                    if scatter_count >= fs_trigger_count:
                        fs_triggered = True
                        fs_triggers_count += 1
                        # Run Free Spins
                        fs_left = fs_award_count
                        while fs_left > 0:
                            fs_screen, _ = self.spin(use_free_reels=True)
                            fs_win_mult, fs_details = self.check_win(fs_screen)
                            fs_total_win += fs_win_mult * total_bet
                            
                            # Add FS stats
                            for d in fs_details:
                                sym = d['symbol']
                                cnt = d['count']
                                sym_win = d['win'] * total_bet
                                symbol_hit_counts[sym][cnt] += 1
                                symbol_win_amounts[sym] += sym_win
                            
                            # Check re-trigger? (Simplified: No re-trigger for now to avoid infinite loops)
                            fs_left -= 1
                
                current_spin_win = (win_multiplier * total_bet) + fs_total_win
                
                if force_win:
                    if current_spin_win > 0:
                        break # Successfully forced a win
                    # Else continue loop to re-spin
                else:
                    break # Normal spin, exit loop
            
            # Update Strategy State
            if current_spin_win > 0:
                current_losing_streak = 0
            else:
                current_losing_streak += 1
            
            # Update Totals
            total_win_amount += current_spin_win
            win_amounts.append(current_spin_win)
            
            # Update Balance
            current_balance = current_balance - total_bet + current_spin_win
            # Record balance every 100 spins to save data size for chart
            if i % 100 == 0:
                balance_history.append(current_balance)
            
            if current_spin_win > 0:
                hits_count += 1
                if current_spin_win > max_win:
                    max_win = current_spin_win
                
                win_ratio = current_spin_win / total_bet
                if win_ratio < 1: bucket = "< 1x Bet"
                elif win_ratio < 5: bucket = "1x - 5x Bet"
                elif win_ratio < 10: bucket = "5x - 10x Bet"
                elif win_ratio < 20: bucket = "10x - 20x Bet"
                elif win_ratio < 50: bucket = "20x - 50x Bet"
                elif win_ratio < 100: bucket = "50x - 100x Bet"
                else: bucket = "100x+ Bet"
                win_dist[bucket] += 1
                
                for d in details:
                    sym = d['symbol']
                    cnt = d['count']
                    # d['win'] is multiplier.
                    sym_win = d['win'] * total_bet
                    
                    symbol_hit_counts[sym][cnt] += 1
                    symbol_win_amounts[sym] += sym_win
            
            if (i + 1) % print_interval == 0:
                print(f"Progress: {i + 1}/{num_spins} spins...")

        rtp = total_win_amount / (num_spins * total_bet)
        hit_rate = hits_count / num_spins
        fs_hit_rate = fs_triggers_count / num_spins if num_spins > 0 else 0
        
        std_dev = np.std(win_amounts)
        # Volatility is usually StdDev of Win Multipliers (Win/Bet).
        # If we use absolute amounts, it scales with Bet.
        # Let's return StdDev of Multipliers for consistency.
        volatility_index = np.std([w / total_bet for w in win_amounts])
        
        margin_of_error = 1.96 * (volatility_index / np.sqrt(num_spins))
        ci_lower = rtp - margin_of_error
        ci_upper = rtp + margin_of_error
        
        # Convert nested dict to regular dict for JSON serialization and ensure keys are native types
        final_symbol_hits = {}
        for sym, counts in symbol_hit_counts.items():
            # sym might be numpy.int64
            native_sym = int(sym)
            final_symbol_hits[native_sym] = {int(k): int(v) for k, v in counts.items()}
            
        final_symbol_win_amounts = {}
        for sym, amount in symbol_win_amounts.items():
            final_symbol_win_amounts[int(sym)] = float(amount)

        return {
            'rtp': float(rtp),
            'total_win': float(total_win_amount),
            'total_bet': float(num_spins * total_bet),
            'total_spins': int(num_spins),
            'hit_rate': float(hit_rate),
            'fs_hit_rate': float(fs_hit_rate),
            'pity_triggers': int(pity_activations_count),
            'max_win': float(max_win),
            'volatility': float(volatility_index),
            'ci_95': (float(ci_lower), float(ci_upper)),
            'symbol_hits': final_symbol_hits,
            'symbol_win_amounts': final_symbol_win_amounts,
            'win_distribution': dict(win_dist),
            'balance_history': [float(b) for b in balance_history]
        }

if __name__ == "__main__":
    import os
    base_dir = '示例卷轴'
    sim = SlotMachine(
        os.path.join(base_dir, 'Payout.xlsx'),
        os.path.join(base_dir, 'SlotNormal.xlsx'),
        os.path.join(base_dir, 'WinLine.xlsx')
    )
    
    print("Starting simulation...")
    results = sim.run_simulation(100000)
    
    output = []
    output.append("="*30)
    output.append(f"Simulation Results ({results['total_spins']} spins)")
    output.append("="*30)
    output.append(f"RTP: {results['rtp'] * 100:.2f}%")
    output.append(f"Volatility (StdDev): {results['volatility']:.4f}")
    output.append(f"95% CI for RTP: [{results['ci_95'][0]*100:.2f}%, {results['ci_95'][1]*100:.2f}%]")
    output.append(f"Hit Rate: {results['hit_rate'] * 100:.2f}%")
    output.append(f"Max Win: {results['max_win']:.2f}")
    output.append(f"Total Win: {results['total_win']:.2f}")
    
    output.append("\nSymbol Stats (Hits 3/4/5 | Total Win | % of Total Win):")
    sorted_syms = sorted(results['symbol_hits'].keys())
    for sym in sorted_syms:
        hits_dict = results['symbol_hits'][sym]
        h3 = hits_dict.get(3, 0)
        h4 = hits_dict.get(4, 0)
        h5 = hits_dict.get(5, 0)
        win = results['symbol_win_amounts'][sym]
        pct = (win / results['total_win']) * 100 if results['total_win'] > 0 else 0
        output.append(f"ID {sym}: 3x:{h3}, 4x:{h4}, 5x:{h5} | {win:.2f} win | {pct:.2f}%")
        
    output.append("\nWin Distribution (Count | % of Spins):")
    order = ["< 1x Bet", "1x - 5x Bet", "5x - 10x Bet", "10x - 20x Bet", "20x - 50x Bet", "50x - 100x Bet", "100x+ Bet"]
    for bucket in order:
        count = results['win_distribution'].get(bucket, 0)
        pct = (count / results['total_spins']) * 100
        output.append(f"{bucket}: {count} | {pct:.2f}%")
        
    print("\n".join(output))
    
    with open('simulation_result.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
    print("\nResults saved to simulation_result.txt")
