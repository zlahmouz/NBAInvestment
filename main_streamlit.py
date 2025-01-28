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
    page_title="NBA Talent Investment Advisor",
    page_icon="📈",
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
    st.subheader("💼 Portfolio d'Indicateurs de Performance")
    with st.form("player_stats"):
        st.markdown("### 📊 Métriques de Production")
        gp = st.number_input("🎮 Taux de Participation (GP)", min_value=0.0, max_value=82.0, value=70.0)
        pts = st.number_input("💫 Rendement Offensif (PTS)", min_value=0.0, max_value=40.0, value=15.0)
        min_played = st.number_input("⏱️ Capital Temps (MIN)", min_value=0.0, max_value=48.0, value=25.0)
        fg = st.number_input("🎯 Indice d'Efficacité des Tirs (FG%)", min_value=0.0, max_value=100.0, value=45.0)
        ft = st.number_input("🎯 Performance sur Coups Francs (FT%)", min_value=0.0, max_value=100.0, value=75.0)
        
        st.markdown("### 📈 Indicateurs de Valeur Ajoutée")
        oreb = st.number_input("💪 Acquisitions Offensives (OREB)", min_value=0.0, max_value=10.0, value=1.5)
        dreb = st.number_input("🛡️ Sécurisation Défensive (DREB)", min_value=0.0, max_value=15.0, value=4.0)
        ast = st.number_input("🤝 Distribution d'Actifs (AST)", min_value=0.0, max_value=15.0, value=3.5)
        stl = st.number_input("💎 Acquisitions Défensives (STL)", min_value=0.0, max_value=5.0, value=1.0)
        blk = st.number_input("🛑 Protection d'Actifs (BLK)", min_value=0.0, max_value=5.0, value=0.5)
        tov = st.number_input("📉 Pertes Opérationnelles (TOV)", min_value=0.1, max_value=10.0, value=2.0)
        
        submitted = st.form_submit_button("📊 Analyser l'Investissement")

if submitted:
    with col2:
        st.subheader("📈 Analyse de l'Investissement")
        
        # Calculs des KPIs
        total_points = gp * pts
        efficiency = pts / min_played
        reb = dreb + oreb
        di = blk + reb + stl
        ast_tov = ast / tov
        oreb_reb = oreb / reb
        game_impact = (ast + reb) / min_played

        input_features = np.array([[fg, ft, total_points, efficiency, di, ast_tov, oreb_reb, game_impact]])
        input_features_scaled = scaler.transform(input_features)
        
        prediction = model.predict(input_features_scaled)
        
        if prediction[0] == 1:
            st.success("""
            💎 Investissement Premium Détecté! 
            
            📈 Les analyses suggèrent un potentiel de rendement à long terme (>5 ans)
            🌟 Recommandation: Position LONG sur cet actif
            """)
            st.balloons()
        else:
            st.warning("""
            ⚠️ Investissement Spéculatif Détecté
            
            📊 Les indicateurs suggèrent un horizon d'investissement court terme (≤5 ans)
            💡 Recommandation: Surveillance active requise, potentiel de plus-value à court terme
            """)
        
        # Visualisations
        st.subheader("📊 Analytics des Performances")
        
        # Graphique radar des KPIs
        categories = ['ROI Offensif 📈', 'Création de Valeur 🤝', 'Sécurité 🛡️', 'Efficience ⚡', 'Impact Global 💫']
        values = [pts/40*10, ast_tov/5*10, di/20*10, efficiency/2*10, game_impact/2*10]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='KPIs de Performance'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="💼 Portfolio de Compétences"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Indice de performance global
        performance_index = (efficiency * 0.3 + ast_tov * 0.2 + di/10 * 0.2 + game_impact * 0.3) / 2 * 100
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = performance_index,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "📈 Indice de Performance Global"},
            gauge = {
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray", 'name': 'Risqué'},
                    {'range': [33, 66], 'color': "gray", 'name': 'Stable'},
                    {'range': [66, 100], 'color': "darkgreen", 'name': 'Premium'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': performance_index
                }
            }))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Graphique de performance comparative
        metrics_df = pd.DataFrame({
            'Indicateur': ['Production Offensive 📈', 'Protection d\'Actifs 🛡️', 'Ratio Valeur/Risque 📊', 'Rendement/Min ⚡'],
            'Valeur': [total_points/2000*10, di/20*10, ast_tov/5*10, game_impact/2*10],
            'Catégorie': ['Offensive', 'Défensive', 'Gestion Risque', 'Efficience']
        })
        
        fig_bars = px.bar(metrics_df, x='Indicateur', y='Valeur', color='Catégorie',
                         title="📊 Analyse Comparative des KPIs (échelle 0-10)")
        fig_bars.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig_bars, use_container_width=True)
        
        # Métriques clés
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1:
            st.metric("📈 Rendement/Minute", f"{efficiency:.2f}")
        with col_metrics2:
            st.metric("🛡️ Indice de Sécurité", f"{di:.1f}")
        with col_metrics3:
            st.metric("📊 Ratio Valeur/Risque", f"{ast_tov:.2f}")

# Sidebar avec informations
st.sidebar.header("💼 Guide d'Investissement")
st.sidebar.info("""
🎯 **Stratégie d'Investissement**

Notre outil analyse les KPIs clés pour évaluer le potentiel ROI d'un talent NBA:

📈 **Métriques de Production**
- Rendement offensif
- Efficacité opérationnelle
- Gestion des actifs

🛡️ **Gestion des Risques**
- Protection d'actifs
- Sécurisation défensive
- Ratio valeur/risque

💎 **Recommandations**
Basées sur l'analyse algorithmique des données historiques NBA
""")
