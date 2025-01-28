import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

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
st.title("📊 NBA Talent Investment Advisor 💎")
st.markdown("""
### 📈 Analysez le ROI potentiel de vos investissements en talents NBA!
Notre outil d'intelligence artificielle évalue le potentiel de longévité des actifs sportifs 
pour maximiser votre portefeuille de talents. 🎯
""")

# Création de colonnes pour une meilleure organisation
col1, col2 = st.columns(2)

# Formulaire de saisie des statistiques
with col1:
    st.subheader("Statistiques du Joueur")
    with st.form("player_stats"):
        gp = st.number_input("Matchs Joués (GP)", min_value=0.0, max_value=80.0, value=70.0)
        pts = st.number_input("Points par Match (PTS)", min_value=0.0, max_value=50.0, value=15.0)
        min_played = st.number_input("Minutes par Match (MIN)", min_value=0.0, max_value=40.0, value=25.0)
        fg = st.number_input("% Pourcentage des Tirs marqués (FG%)", min_value=0.0, max_value=100.0, value=45.0)
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
        st.subheader("📈 Analyse de l'Investissement")
        
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
        
        # Visualisations des statistiques avancées
        st.subheader("Visualisations des statistiques avancées")
        
        # 1. Graphique radar des statistiques principales
        categories = ['Scoring', 'Playmaking', 'Defense', 'Efficiency', 'Impact']
        values = [pts/40*10, ast_tov/5*10, di/20*10, efficiency/2*10, game_impact/2*10]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Stats du Joueur'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Profil du Joueur"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # 2. Jauge pour l'efficacité globale
        overall_impact = (efficiency * 0.3 + ast_tov * 0.2 + di/10 * 0.2 + game_impact * 0.3) / 2 * 100
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = overall_impact,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Impact Global"},
            gauge = {'axis': {'range': [None, 100]},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkblue"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': overall_impact}}))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 3. Graphique en barres des statistiques avancées
        stats_df = pd.DataFrame({
            'Statistique': ['Total Points', 'Impact Défensif', 'Ratio AST/TOV', 'Impact/Min'],
            'Valeur': [total_points/2000*10, di/20*10, ast_tov/5*10, game_impact/2*10],
            'Catégorie': ['Scoring', 'Defense', 'Playmaking', 'Overall']
        })
        
        fig_bars = px.bar(stats_df, x='Statistique', y='Valeur', color='Catégorie',
                         title="Comparaison des Statistiques Avancées (échelle 0-10)")
        fig_bars.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig_bars, use_container_width=True)
        
        # Métriques simples
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Points par Minute", f"{efficiency:.2f}")
        with col_stats2:
            st.metric("Impact Défensif", f"{di:.1f}")
        with col_stats3:
            st.metric("Ratio AST/TOV", f"{ast_tov:.2f}")

# Ajout d'informations supplémentaires
st.sidebar.header("À propos")
st.sidebar.info("""
Cette application utilise un modèle de Machine Learning entraîné sur des données historiques de la NBA
pour prédire la longévité de la carrière d'un joueur.
""")

st.sidebar.header("Guide des Visualisations")
st.sidebar.markdown("""
1. **Graphique Radar**: Montre le profil global du joueur sur 5 aspects clés
2. **Jauge d'Impact**: Mesure l'impact global du joueur sur une échelle de 0 à 100
3. **Graphique en Barres**: Compare les différentes statistiques avancées
""")
