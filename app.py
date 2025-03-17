from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
from waitress import serve
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union, Optional

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition des chemins des fichiers Excel
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

FILE_PATHS = {
    'compatibility': os.path.join(UPLOAD_FOLDER, 'PARTS_COMPATIBILITY_REPORT.xlsx'),
    'stock': os.path.join(UPLOAD_FOLDER, 'Stock.xlsx')
}

# Fonction pour remplacer les NaN par None dans les DataFrames
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace toutes les valeurs NaN dans un DataFrame par None.
    """
    return df.replace({np.nan: None})

# Fonction pour remplacer les NaN par None dans les structures de données
def replace_nan(data: Union[dict, list, float]) -> Union[dict, list, None]:
    """
    Remplace les valeurs NaN par None pour éviter les problèmes de sérialisation JSON.
    """
    if isinstance(data, dict):
        return {k: replace_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan(item) for item in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    return data

# Chargement des fichiers Excel en DataFrames pandas
def load_data(file_paths: Dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les fichiers Excel et nettoie les données.
    Si les fichiers n'existent pas, crée des DataFrames vides.
    """
    dfs = {}
    for key, path in file_paths.items():
        if os.path.exists(path):
            dfs[key] = pd.read_excel(path)
        else:
            # Crée un DataFrame vide avec les colonnes nécessaires
            if key == 'compatibility':
                dfs[key] = pd.DataFrame(columns=['PLATFORM', 'PART NUMBER', 'SUBS NUMBER'])
            elif key == 'stock':
                dfs[key] = pd.DataFrame(columns=['AR_Ref', 'AR_Design', 'DE_Intitule'])
            logger.warning(f"Fichier Excel introuvable : {path}. Un DataFrame vide a été créé.")

    df1 = dfs.get('compatibility', pd.DataFrame())
    df2 = dfs.get('stock', pd.DataFrame())

    # Nettoyage et normalisation des colonnes
    if not df1.empty:
        df1 = df1.dropna(subset=['PLATFORM'])  # Supprime les lignes avec PLATFORM vide
        df1['PLATFORM'] = df1['PLATFORM'].str.lower().str.strip()
        df1['PART NUMBER'] = df1['PART NUMBER'].astype(str).str.strip()

    if not df2.empty:
        df2['AR_Ref'] = df2['AR_Ref'].astype(str).str.strip()
        df2['AR_Design'] = df2['AR_Design'].astype(str).str.strip()

    # Vérification des colonnes requises dans df2
    required_columns = {'AR_Ref', 'AR_Design', 'DE_Intitule'}
    if not required_columns.issubset(df2.columns):
        raise ValueError(f"Les colonnes {required_columns} sont manquantes dans df2.")

    # Remplacement des NaN par None dans les DataFrames
    df1 = clean_dataframe(df1)
    df2 = clean_dataframe(df2)

    logger.info("Fichiers Excel chargés et nettoyés avec succès.")
    return df1, df2

# Chargement des données au démarrage de l'application
try:
    df1, df2 = load_data(FILE_PATHS)
except Exception as e:
    logger.error(f"Erreur lors du chargement des fichiers Excel: {e}")
    df1, df2 = pd.DataFrame(), pd.DataFrame()  # Crée des DataFrames vides en cas d'erreur

# Route pour importer un fichier Excel
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Importe un fichier Excel et le sauvegarde dans le dossier uploads.
    """
    if 'file' not in request.files:
        return jsonify({"message": "Aucun fichier fourni"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "Aucun fichier sélectionné"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Mettre à jour les chemins des fichiers
    if 'PARTS_COMPATIBILITY_REPORT' in file.filename:
        FILE_PATHS['compatibility'] = file_path
    elif 'Stock' in file.filename:
        FILE_PATHS['stock'] = file_path

    # Recharger les données après l'importation
    global df1, df2
    try:
        df1, df2 = load_data(FILE_PATHS)
    except Exception as e:
        logger.error(f"Erreur lors du rechargement des fichiers Excel: {e}")
        return jsonify({"message": "Erreur lors du rechargement des fichiers Excel"}), 500

    return jsonify({"message": "Fichier importé avec succès", "file_path": file_path}), 200

# Route pour supprimer un fichier Excel
@app.route('/delete', methods=['POST'])
def delete_file():
    """
    Supprime un fichier Excel du dossier uploads.
    """
    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({"message": "Aucun chemin de fichier fourni"}), 400

    file_path = data['file_path']
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": "Fichier supprimé avec succès"}), 200
    else:
        return jsonify({"message": "Fichier introuvable"}), 404

# Route pour mettre à jour les fichiers Excel
@app.route('/update-files', methods=['POST'])
def update_files():
    """
    Met à jour les fichiers Excel en les remplaçant par de nouveaux fichiers.
    """
    if 'file' not in request.files:
        return jsonify({"message": "Aucun fichier fourni"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "Aucun fichier sélectionné"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Mettre à jour les chemins des fichiers
    if 'PARTS_COMPATIBILITY_REPORT' in file.filename:
        FILE_PATHS['compatibility'] = file_path
    elif 'Stock' in file.filename:
        FILE_PATHS['stock'] = file_path

    # Recharger les données après la mise à jour
    global df1, df2
    try:
        df1, df2 = load_data(FILE_PATHS)
    except Exception as e:
        logger.error(f"Erreur lors du rechargement des fichiers Excel: {e}")
        return jsonify({"message": "Erreur lors du rechargement des fichiers Excel"}), 500

    return jsonify({"message": "Fichiers mis à jour avec succès", "file_path": file_path}), 200

# Route pour vérifier si l'API fonctionne
@app.route('/health', methods=['GET'])
def health_check():
    """
    Vérifie que l'API est fonctionnelle.
    """
    return jsonify({"status": "ok"}), 200

# Fonction pour trouver des correspondances approximatives avec RapidFuzz
def improved_fuzzy_match(query: str, choices: List[str], limit: int = 3) -> List[tuple[str, float]]:
    """
    Trouve des correspondances approximatives pour une requête donnée en utilisant RapidFuzz.
    """
    matches = process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=limit)
    return [(match[0], match[1] / 100) for match in matches]  # Normalisation du score entre 0 et 1

# Route de recherche
@app.route('/search', methods=['POST'])
def search():
    """
    Recherche une plateforme et un numéro de pièce dans les fichiers Excel.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"message": "Aucune donnée fournie"}), 400

        platform = data.get('platform', '').lower().strip()
        pn = data.get('pn', '').strip()

        if not platform and not pn:
            return jsonify({"message": "Au moins une plateforme ou un numéro de pièce est requis"}), 400

        logger.info(f"Recherche en cours : Plateforme='{platform}', Part Number='{pn}'")

        # Initialisation des résultats
        results_df1 = df1.copy()
        results_df2 = df2.copy()
        tableau_compatibilite = []
        tableau_subs = []
        tableau_stock = []

        # Filtrage par plateforme
        if platform:
            results_df1 = results_df1[results_df1['PLATFORM'].str.contains(platform, case=False, na=False)]
            results_df2 = results_df2[results_df2['AR_Design'].str.contains(platform, case=False, na=False)]

        # Filtrage par numéro de pièce
        if pn:
            results_df1 = results_df1[results_df1['PART NUMBER'].str.contains(pn, case=False, na=False)]
            results_df2 = results_df2[results_df2['AR_Ref'].str.contains(pn, case=False, na=False)]

        # Si des résultats sont trouvés dans df1
        if not results_df1.empty:
            subs_numbers = results_df1['SUBS NUMBER'].dropna().unique()
            for subs in subs_numbers:
                subs_matches = df2[
                    (df2['AR_Ref'].str.contains(subs, case=False, na=False)) &
                    (df2['DE_Intitule'].isin(['Dépôt 2', 'MAQUETTE']))
                ]
                if not subs_matches.empty:
                    tableau_subs.extend(replace_nan(subs_matches.to_dict(orient='records')))

            tableau_compatibilite = replace_nan(results_df1.to_dict(orient='records'))

        # Si des résultats sont trouvés dans df2
        if not results_df2.empty:
            results_df2 = results_df2[results_df2['DE_Intitule'].isin(['Dépôt 2', 'MAQUETTE'])]
            tableau_stock = replace_nan(results_df2.to_dict(orient='records'))

        # Si aucun résultat exact, utiliser fuzzy matching pour suggestions
        suggestions = {}
        if (platform and results_df1.empty and results_df2.empty) or (pn and results_df1.empty and results_df2.empty):
            if platform:
                suggestions["platform_suggestions"] = improved_fuzzy_match(platform, df1['PLATFORM'].unique())
            if pn:
                suggestions["pn_suggestions"] = improved_fuzzy_match(pn, df1['PART NUMBER'].unique())

        # Réponse finale
        response = {
            "message": f"Résultats trouvés pour la plateforme '{platform}' et le numéro de pièce '{pn}'.",
            "tableau_compatibilite": tableau_compatibilite,
            "tableau_subs": tableau_subs,
            "tableau_stock": tableau_stock,
            "suggestions": suggestions if suggestions else None
        }
        logger.debug(f"Réponse avant sérialisation : {response}")
        if not (tableau_compatibilite or tableau_stock):
            response["message"] = "Non Compatible"
            logger.warning(f"Aucun résultat exact trouvé pour PN '{pn}' et plateforme '{platform}'")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        return jsonify({"message": f"Erreur serveur: {str(e)}"}), 500

# Lancement de l'application Flask
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    serve(app, host='0.0.0.0', port=port)