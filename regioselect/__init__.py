import hashlib
import threading

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import PrepareMolForDrawing
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(False)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
import os

from regioselect.scripts.src.predictor import run_predictions

app = Flask(__name__)
app.config.from_object(__name__)

# app.config['DEBUG'] = False

# Database name
db_name = 'data/regioselect.db'
basedir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(basedir, db_name)

# Configure the SQLite database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialize the app with Flask-SQLAlchemy
db = SQLAlchemy(app)


class regioselect_results(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hash_code = db.Column(db.String(64), unique=True, nullable=False)
    rdkit_smiles = db.Column(db.String(400), unique=True, nullable=False)
    
    # types of status: none (not submitted), error (calc failed e.g. due to timeout), pending (calc in queue or running), complete (calc done)
    ml_status = db.Column(db.String(10), default='none')
    xtb_status = db.Column(db.String(10), default='none')
    dft_sp_status = db.Column(db.String(10), default='none')
    dft_opt_status = db.Column(db.String(10), default='none')

    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f'<Molecule: {self.rdkit_smiles} ({self.hash_code})>'
    

def perform_predictions(hash_code, smiles):
    with app.app_context():
        try:
            # Run of machine learning predictions
            run_predictions(smiles=smiles, name=hash_code)

            # Update the status to 'complete' in the database
            regioselect_res = regioselect_results.query.filter_by(hash_code=hash_code).first()
            regioselect_res.ml_status = 'complete'
            db.session.commit()

        except Exception as e:
            # Update the status to 'error' in the database
            regioselect_res = regioselect_results.query.filter_by(hash_code=hash_code).first()
            regioselect_res.ml_status = 'error'
            db.session.commit()       


@app.route('/data/desc_calcs/<path:filepath>')
def data_pred(filepath):
    return send_from_directory('data/desc_calcs', filepath)


@app.route('/check_MLresults/<hash_code>')
def check_MLresults(hash_code):
    regioselect_res = regioselect_results.query.filter_by(hash_code=hash_code).first()
    return jsonify({'status': regioselect_res.ml_status})
            

@app.route('/', methods=['GET', 'POST'])
def index():
   
    if request.method == 'POST':
        smiles = request.form['smiles']

        if smiles == '':
            return render_template('index.html', error='Please provide an input SMILES.')
        elif len(smiles.split('.')) > 1:
            return render_template('index.html', error='You cannot submit more than one molecule at a time.')
        
        try:
            # Canonicalize input smiles
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

            # Convert the SMILES string to a hash code (MD5)
            hash_object = hashlib.md5(smiles.encode())
            hash_code = hash_object.hexdigest()

            # Check if the hash_code already exists in the database
            regioselect_res = regioselect_results.query.filter_by(hash_code=hash_code).first()
        except Exception as e:
            print(str(e))
            return render_template('index.html', error='Error processing the input SMILES.')
        
        # Check the number of heavy atoms in the submitted molecule (limit is set to 30 HAs)
        if Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() > 30:
            return render_template('index.html', error=f'The number of heavy atoms ({Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()}) exceeds the limit of 30 for ML!')
        
        if not regioselect_res:
            # Start predictions
            prediction_thread = threading.Thread(target=perform_predictions, args=(hash_code, smiles))
            prediction_thread.start()

            # Add molecule and set the ml_status to 'pending' in the database
            db.session.add(regioselect_results(hash_code=hash_code, rdkit_smiles=smiles, ml_status='pending'))
            db.session.commit()
        elif regioselect_res.ml_status == 'none': # use "else" to rerun all calculations 
            # Start predictions
            prediction_thread = threading.Thread(target=perform_predictions, args=(hash_code, smiles))
            prediction_thread.start()
            
            # Update the status to 'pending' in the database
            regioselect_res.ml_status = 'pending'
            db.session.commit()
            
        return redirect(url_for('MLresults', hash_code=hash_code))

    return render_template('index.html')


@app.route('/MLresults/<hash_code>', methods=['GET'])
def MLresults(hash_code):

    # Retrieve data associated with the hash code
    regioselect_res = regioselect_results.query.filter_by(hash_code=hash_code).first()

    if not regioselect_res or regioselect_res.ml_status == 'none':
    
        return render_template('index.html', error='Results are not found.')
    
    elif regioselect_res.ml_status == 'error':

        return render_template('index.html', error=f'Prediction ERROR! Please submit an issue on GitHub with ID: {hash_code}')
    
    elif regioselect_res.ml_status == 'pending':
    
        return render_template('MLresults.html', regioselect_res=regioselect_res)
    
    elif regioselect_res.ml_status == 'complete':
        df_eas = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/df_eas_{hash_code}.pkl')
        df_bde = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/df_bde_{hash_code}.pkl')[['Atom ID', 'BDE Value [kcal/mol]', 'BDFE Value [kcal/mol]', 'Reactant', '%Vbur']]
        df_pka = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/df_pka_{hash_code}.pkl')
        df_ha = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/df_ha_{hash_code}.pkl')
        df_nuc = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/df_nuc_{hash_code}.pkl')
        df_elec = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/df_elec_{hash_code}.pkl')
        df_steric = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/df_steric_{hash_code}.pkl')
        table_eas = df_eas.to_html(index=False, decimal='.', float_format='%.2f', escape=False).replace('<table border="1" class="dataframe">', '<table>\n<caption class="caption">Electrophilic Aromatic Substitution (EAS)</caption>').replace('<tr style="text-align: right;">', '<tr>')
        table_bde = df_bde.to_html(index=False, decimal='.', float_format='%.2f', escape=False).replace('<table border="1" class="dataframe">', '<table>\n<caption class="caption">C-H Bond Dissociation Energy (BDE) </caption>').replace('<tr style="text-align: right;">', '<tr>')
        table_pka = df_pka.to_html(index=False, decimal='.', float_format='%.2f', escape=False).replace('<table border="1" class="dataframe">', '<table>\n<caption class="caption">C-H pKa</caption>').replace('<tr style="text-align: right;">', '<tr>')
        table_ha = df_ha.to_html(index=False, decimal='.', float_format='%.2f', escape=False).replace('<table border="1" class="dataframe">', '<table>\n<caption class="caption">Hydride Affinity (HA)</caption>').replace('<tr style="text-align: right;">', '<tr>')
        table_nuc = df_nuc.to_html(index=False, decimal='.', float_format='%.2f', escape=False).replace('<table border="1" class="dataframe">', '<table>\n<caption class="caption">Nucleophilicity | Methyl Cation Affinity (MCA)</caption>').replace('<tr style="text-align: right;">', '<tr>')
        table_elec = df_elec.to_html(index=False, decimal='.', float_format='%.2f', escape=False).replace('<table border="1" class="dataframe">', '<table>\n<caption class="caption">Electrophilicity | Methyl Anion Affinity (MAA)</caption>').replace('<tr style="text-align: right;">', '<tr>')
        table_steric = df_steric.to_html(index=False, decimal='.', float_format='%.2f', escape=False).replace('<table border="1" class="dataframe">', '<table>\n<caption class="caption">Sterics </caption>').replace('<tr style="text-align: right;">', '<tr>')
        return render_template('MLresults.html', regioselect_res=regioselect_res, table_eas=table_eas, table_bde=table_bde, table_pka=table_pka, table_ha=table_ha, table_nuc=table_nuc, table_elec=table_elec, table_steric=table_steric)


@app.route('/smiles_to_image')
def smiles_to_image():
    # Extract sites to be highlighted
    smiles = request.args.get('smiles', type=str)
    print(smiles, flush=True)
    hash_code = request.args.get('hash_code', type=str)
    table_name = request.args.get('table_name', type=str)
    df = pd.read_pickle(f'regioselect/data/desc_calcs/{hash_code}/{table_name}_{hash_code}.pkl')

    if table_name == 'df_eas':
        sites = [int(x) for x in df[df['EAS Score [%]'] >= 50]['Atom ID'].tolist()]

    elif table_name == 'df_steric':
        value_name = f'% buried volume'
        # No highlighting for the sterics!
        sites = [] 

    else:
        value_name = [x for x in df.keys() if 'Value' in x][0]
        if table_name in ['df_nuc', 'df_elec']:
            sites = [int(x) for x in df[df[value_name] >= df[value_name].max() - 12.6]['Atom ID'].tolist()] # hightlight everything within cutoff of 12.6 kJ/mol
        else:
            sites = [int(x) for x in df[df[value_name] <= df[value_name].min()]['Atom ID'].tolist()]
    # Drawing code
    d2d = rdMolDraw2D.MolDraw2DSVG(350, 350)
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    dos.atomHighlightsAreCircles = False
    dos.fillHighlights = True
    dos.addAtomIndices = True
    dos.minFontSize = 18
    dos.annotationFontScale = 0.85
    
    atomHighlighs = {}
    highlightRads = {}
    for site in sites:
        atomHighlighs[site] = [(1.0, 0.0, 0.0, 0.25)] # light red color
        highlightRads[site] = 0.35
    
    rdkit_mol = PrepareMolForDrawing(Chem.MolFromSmiles(smiles))
    d2d.DrawMoleculeWithHighlights(rdkit_mol, '', dict(atomHighlighs), {}, highlightRads, {})
    d2d.FinishDrawing()
    return Response(d2d.GetDrawingText(), mimetype='image/svg+xml')


@app.route('/MLviewer/<path:sdf_path>', methods=['GET'])
def MLviewer(sdf_path):

    sdf_path = '/data/desc_calcs/' + sdf_path
    print(sdf_path)

    # Render an HTML template for the pop-up windowregioselect/data/desc_calcs/b5bd73a800ba4c220883457154ae2e80
    return render_template('MLjsmol.html', sdf_path=sdf_path)


if __name__ == '__main__':
    app.run(debug=True)

