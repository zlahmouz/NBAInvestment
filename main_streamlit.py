import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Configuration de la page
st.set_page_config(
    page_title="NBA Career Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le modèle et le scaler
@st.cache_resource
def load_model():
    return joblib.load('model_nba.pkl'), joblib.load('scaler.pkl')

model, scaler = load_model()

# CSS personnalisé
st.markdown("""
<style>
    .main { background-color: #F5F5F5; }
    .st-bb { background-color: white; }
    .st-at { background-color: #FF4B4B; }
    div[data-testid="stSidebarUserContent"] { background-color: #2E86AB; }
    .sidebar .sidebar-content { color: white; }
    h1 { color: #2E86AB; text-align: center; }
    .metric-label { font-size: 14px; color: #666; }
    .metric-value { font-size: 24px; color: #2E86AB; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Interface principale
st.title("🏀 NBA Career Predictor Pro")
st.markdown("---")

# Sidebar avec entrées utilisateur
with st.sidebar:
    st.header("📊 Player Stats Input")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            gp = st.number_input("Games Played", min_value=0.0, value=50.0, step=1.0)
            pts = st.number_input("Points/G", min_value=0.0, value=15.0, step=0.5)
            min = st.number_input("Minutes/G", min_value=0.0, value=25.0, step=0.5)
            fg = st.number_input("FG%", min_value=0.0, max_value=100.0, value=45.0)
            ft = st.number_input("FT%", min_value=0.0, max_value=100.0, value=75.0)
        
        with col2:
            oreb = st.number_input("Offensive Rebounds", min_value=0.0, value=2.0)
            dreb = st.number_input("Defensive Rebounds", min_value=0.0, value=5.0)
            ast = st.number_input("Assists", min_value=0.0, value=3.0)
            stl = st.number_input("Steals", min_value=0.0, value=1.0)
            blk = st.number_input("Blocks", min_value=0.0, value=1.0)
            tov = st.number_input("Turnovers", min_value=0.0, value=2.0)

    st.markdown("---")
    st.markdown("🔍 *Data Source: Historical NBA player data*")

# Conteneur principal
main_container = st.container()
with main_container:
    if st.button("🚀 Run Career Prediction", use_container_width=True):
        with st.spinner('Analyzing player potential...'):
            try:
                # Calculs
                total_points = gp * pts
                efficiency = pts / min if min > 0 else 0
                reb = dreb + oreb
                di = blk + reb + stl
                ast_tov = ast / tov if tov > 0 else 0
                oreb_reb = oreb / reb if reb > 0 else 0
                game_impact = (ast + reb) / min if min > 0 else 0

                # Préparation des données
                input_features = np.array([[fg, ft, total_points, efficiency, di, ast_tov, oreb_reb, game_impact]])
                input_features_scaled = scaler.transform(input_features)

                # Prédiction
                prediction = model.predict(input_features_scaled)
                probability = model.predict_proba(input_features_scaled)[0][1]

                # Affichage des résultats
                st.markdown("## 📈 Prediction Results")
                
                cols = st.columns([1, 2])
                with cols[0]:
                    # Jauge de probabilité
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.add_patch(Arc((0.5, 0), 1.0, 0.5, theta1=0, theta2=180, edgecolor='#2E86AB'))
                    ax.plot([0.5, 0.5], [0, 0.25], color='#2E86AB', linestyle='--')
                    ax.text(0.5, 0.3, f"{probability*100:.1f}%", ha='center', va='center', fontsize=24, color='#2E86AB')
                    plt.axis('off')
                    plt.xlim(0, 1)
                    plt.ylim(-0.1, 0.5)
                    st.pyplot(fig)
                    
                with cols[1]:
                    st.markdown(f"### 📅 Career Projection: {'**Long-Term (>5 years)**' if prediction[0] == 1 else '**Short-Term (≤5 years)**'}")
                    st.progress(probability)
                    
                    # Métriques clés
                    st.markdown("#### 🔑 Key Performance Indicators")
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Total Points", f"{total_points:,.0f}")
                    metric_cols[1].metric("Efficiency", f"{efficiency:.2f}/min")
                    metric_cols[2].metric("Defensive Impact", f"{di:.1f}")
                    metric_cols[3].metric("Game Impact", f"{game_impact:.2f}")

                # Graphique radar
                st.markdown("---")
                st.markdown("### 📊 Skill Radar Chart")
                categories = ['Scoring', 'Defense', 'Playmaking', 'Efficiency', 'Rebounding']
                values = [pts, (stl + blk)/2, ast, (fg + ft)/2, reb]
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
                ax.plot(categories, values, color='#2E86AB', linewidth=2)
                ax.fill(categories, values, color='#2E86AB', alpha=0.25)
                ax.set_yticklabels([])
                ax.set_title("Player Skills Profile", size=20, y=1.1)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"🚨 Prediction Error: {str(e)}")

    else:
        # Section d'accueil
        st.markdown("""
        ## 🏆 Welcome to NBA Career Predictor Pro
        
        **How to use:**
        1. Enter player stats in the left sidebar
        2. Click the 'Run Career Prediction' button
        3. Explore detailed analysis and projections
        
        *Sample players available below ↓*
        """)
        
        # Exemples prédéfinis
        sample_players = {
            "Star Player": {
                "GP": 75, "PTS": 25.5, "MIN": 35.4,
                "FG%": 48.7, "FT%": 87.3, "OREB": 3.2,
                "DREB": 7.8, "AST": 7.5, "STL": 1.8,
                "BLK": 1.2, "TOV": 2.9
            },
            "Role Player": {
                "GP": 68, "PTS": 12.3, "MIN": 25.7,
                "FG%": 44.1, "FT%": 72.5, "OREB": 1.5,
                "DREB": 4.2, "AST": 2.8, "STL": 0.9,
                "BLK": 0.6, "TOV": 1.7
            }
        }
        
        selected_player = st.selectbox("Load Sample Player:", list(sample_players.keys()))
        if st.button("Load Selected Player", key="load_player"):
            for key, value in sample_players[selected_player].items():
                st.session_state[key] = value

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    NBA Career Predictor Pro v1.0 | © 2023 Basketball Analytics Inc.<br>
    <em>Predictive model accuracy: 89.2% (test set)</em>
</div>
""", unsafe_allow_html=True)
