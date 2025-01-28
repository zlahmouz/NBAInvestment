import streamlit as st
import joblib
import numpy as np

# Charger le modèle ML et le scaler pré-entraînés
@st.cache_resource
def load_model():
    model = joblib.load('model_nba.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur de Carrière NBA",
    page_icon="🏀",
    layout="wide"
)

# Titre et description
st.title("🏀 Prédicteur de Carrière NBA")
st.markdown("""
Cette application utilise le Machine Learning pour prédire si un joueur NBA aura une carrière supérieure à 5 ans 
en se basant sur ses statistiques.
""")

# Création de colonnes pour une meilleure organisation
col1, col2 = st.columns(2)

# Formulaire de saisie des statistiques
with col1:
    st.subheader("Statistiques du Joueur")
    with st.form("player_stats"):
        gp = st.number_input("Matchs Joués (GP)", min_value=0.0, max_value=82.0, value=70.0)
        pts = st.number_input("Points par Match (PTS)", min_value=0.0, max_value=40.0, value=15.0)
        min_played = st.number_input("Minutes par Match (MIN)", min_value=0.0, max_value=48.0, value=25.0)
        fg = st.number_input("% Tirs (FG%)", min_value=0.0, max_value=100.0, value=45.0)
        ft = st.number_input("% Lancers Francs (FT%)", min_value=0.0, max_value=100.0, value=75.0)
        
        # Deuxième colonne de statistiques
        oreb = st.number_input("Rebonds Offensifs (OREB)", min_value=0.0, max_value=10.0, value=1.5)
        dreb = st.number_input("Rebonds Défensifs (DREB)", min_value=0.0, max_value=15.0, value=4.0)
        ast = st.number_input("Passes Décisives (AST)", min_value=0.0, max_value=15.0, value=3.5)
        stl = st.number_input("Interceptions (STL)", min_value=0.0, max_value=5.0, value=1.0)
        blk = st.number_input("Contres (BLK)", min_value=0.0, max_value=5.0, value=0.5)
        tov = st.number_input("Pertes de Balle (TOV)", min_value=0.1, max_value=10.0, value=2.0)
        
        submitted = st.form_submit_button("Prédire")

# Calculs et prédiction
if submitted:
    with col2:
        st.subheader("Résultats de l'Analyse")
        
        # Calcul des statistiques avancées
        total_points = gp * pts
        efficiency = pts / min_played
        reb = dreb + oreb
        di = blk + reb + stl
        ast_tov = ast / tov
        oreb_reb = oreb / reb
        game_impact = (ast + reb) / min_played

        # Préparation des features pour la prédiction
        input_features = np.array([[fg, ft, total_points, efficiency, di, ast_tov, oreb_reb, game_impact]])
        input_features_scaled = scaler.transform(input_features)
        
        # Prédiction
        prediction = model.predict(input_features_scaled)
        
        # Affichage du résultat avec mise en forme
        if prediction[0] == 1:
            st.success("🌟 Prédiction : Carrière > 5 ans")
            st.balloons()
        else:
            st.warning("⚠️ Prédiction : Carrière ≤ 5 ans")
        
        # Affichage des statistiques avancées calculées
        st.subheader("Statistiques Avancées")
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.metric("Total Points sur la Saison", f"{total_points:.1f}")
            st.metric("Efficacité Scoring", f"{efficiency:.2f} pts/min")
            st.metric("Impact Défensif", f"{di:.1f}")
            
        with col_stats2:
            st.metric("Ratio AST/TOV", f"{ast_tov:.2f}")
            st.metric("% Rebonds Offensifs", f"{oreb_reb*100:.1f}%")
            st.metric("Impact par Minute", f"{game_impact:.2f}")

# Ajout d'informations supplémentaires
st.sidebar.header("À propos")
st.sidebar.info("""
Cette application utilise un modèle de Machine Learning entraîné sur des données historiques de la NBA
pour prédire la longévité de la carrière d'un joueur.
""")

st.sidebar.header("Comment utiliser")
st.sidebar.markdown("""
1. Entrez les statistiques du joueur dans les champs fournis
2. Cliquez sur 'Prédire'
3. Examinez les résultats et les statistiques avancées calculées
""")
