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
        
        wild_id = None
        if wild_id_str:
            try:
                wild_id = int(wild_id_str)
            except:
                pass

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
            
            if len(file_paths) < 3:
                 return jsonify({'error': 'Missing required files. Please upload Payout.xlsx, SlotNormal.xlsx, and WinLine.xlsx'}), 400

            current_sim = SlotMachine(
                file_paths['payout'],
                file_paths['reel'],
                file_paths['line'],
                wild_id=wild_id
            )
            
            if num_spins > 1000000: num_spins = 1000000
            
            results = current_sim.run_simulation(num_spins, total_bet=100.0)
            return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
