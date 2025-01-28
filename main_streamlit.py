import streamlit as st
import joblib
import numpy as np

# Charger le modèle et le scaler
model = joblib.load('model_nba.pkl')
scaler = joblib.load('scaler.pkl')

# Configuration de la page
st.set_page_config(page_title="NBA Career Predictor", layout="wide")

# Interface utilisateur
st.title("Prédiction de durée de carrière NBA 🏀")
st.markdown("""
Cette application prédit si un joueur NBA aura une carrière de plus de 5 ans
en fonction de ses statistiques.
""")

# Création des inputs utilisateur dans une sidebar
with st.sidebar:
    st.header("📊 Entrez les statistiques du joueur")
    
    gp = st.number_input("Matchs joués (GP)", min_value=0.0, value=50.0)
    pts = st.number_input("Points par match (PTS)", min_value=0.0, value=10.0)
    min = st.number_input("Minutes par match (MIN)", min_value=0.0, value=20.0)
    fg = st.number_input("Pourcentage de tirs réussis (FG%)", min_value=0.0, max_value=100.0, value=45.0)
    ft = st.number_input("Pourcentage de lancers francs (FT%)", min_value=0.0, max_value=100.0, value=75.0)
    oreb = st.number_input("Rebonds offensifs (OREB)", min_value=0.0, value=2.0)
    dreb = st.number_input("Rebonds défensifs (DREB)", min_value=0.0, value=5.0)
    ast = st.number_input("Passes décisives (AST)", min_value=0.0, value=3.0)
    stl = st.number_input("Interceptions (STL)", min_value=0.0, value=1.0)
    blk = st.number_input("Contres (BLK)", min_value=0.0, value=1.0)
    tov = st.number_input("Pertes de balle (TOV)", min_value=0.0, value=2.0)

# Calcul des caractéristiques dérivées
if st.button("Prédire la durée de carrière"):
    try:
        total_points = gp * pts
        efficiency = pts / min if min > 0 else 0
        reb = dreb + oreb
        di = blk + reb + stl
        ast_tov = ast / tov if tov > 0 else 0
        oreb_reb = oreb / reb if reb > 0 else 0
        game_impact = (ast + reb) / min if min > 0 else 0

        # Préparation des features pour le modèle
        input_features = np.array([[fg, ft, total_points, efficiency, di, ast_tov, oreb_reb, game_impact]])
        input_features_scaled = scaler.transform(input_features)

        # Prédiction
        prediction = model.predict(input_features_scaled)
        result = 'Carrière > 5 ans' if prediction[0] == 1 else 'Carrière <= 5 ans'

        # Affichage du résultat
        st.success(f"Résultat de la prédiction: {result}")
        st.balloons()
        
    except Exception as e:
        st.error(f"Une erreur est survenue: {str(e)}")

# Section d'explications
with st.expander("ℹ️ Comment utiliser cette application"):
    st.markdown("""
    1. Remplissez toutes les statistiques dans la barre latérale
    2. Cliquez sur le bouton 'Prédire la durée de carrière'
    3. Consultez le résultat en haut de la page
    """)

# Note sur les données
st.caption("Note: Le modèle utilise des données historiques de joueurs NBA pour ses prédictions.")
