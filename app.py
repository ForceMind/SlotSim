from flask import Flask, render_template, jsonify, request
import os
import tempfile
from slot_machine import SlotMachine

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        # Handle FormData
        num_spins = int(request.form.get('num_spins', 10000))
        wild_id_str = request.form.get('wild_id', '')
        
        # Free Spin Params
        scatter_id_str = request.form.get('scatter_id', '')
        fs_trigger_count = int(request.form.get('fs_trigger_count', 3))
        fs_award_count = int(request.form.get('fs_award_count', 10))
        
        # Strategy Params
        pity_streak = int(request.form.get('pity_streak', 0))
        
        wild_id = None
        if wild_id_str:
            try: wild_id = int(wild_id_str)
            except: pass
            
        scatter_id = None
        if scatter_id_str:
            try: scatter_id = int(scatter_id_str)
            except: pass

        uploaded_files = request.files.getlist('files[]')
        if not uploaded_files:
             return jsonify({'error': 'No files uploaded'}), 400

        # Create temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = {}
            for file in uploaded_files:
                filename = file.filename
                path = os.path.join(temp_dir, filename)
                file.save(path)
                
                # Simple heuristic to identify files
                if 'Payout' in filename: file_paths['payout'] = path
                elif 'SlotNormal' in filename: file_paths['reel'] = path
                elif 'WinLine' in filename: file_paths['line'] = path
                elif 'SlotFree' in filename: file_paths['free_reel'] = path
            
            if len(file_paths) < 3:
                 return jsonify({'error': 'Missing required files. Please upload Payout.xlsx, SlotNormal.xlsx, and WinLine.xlsx'}), 400

            current_sim = SlotMachine(
                file_paths['payout'],
                file_paths['reel'],
                file_paths['line'],
                wild_id=wild_id,
                free_spin_reel_path=file_paths.get('free_reel')
            )
            
            if num_spins > 1000000: num_spins = 1000000
            
            results = current_sim.run_simulation(
                num_spins, 
                total_bet=100.0,
                scatter_id=scatter_id,
                fs_trigger_count=fs_trigger_count,
                fs_award_count=fs_award_count,
                pity_streak_threshold=pity_streak
            )
            return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # host='0.0.0.0' allows access from other computers in the same network
    app.run(debug=True, host='0.0.0.0', port=5000)
