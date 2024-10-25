from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle ML et le scaler pré-entraînés
model = joblib.load('model_nba.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')  # Formulaire HTML pour soumettre les caractéristiques du joueur


@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les données du formulaire HTML
    gp = float(request.form['GP'])
    pts = float(request.form['PTS'])
    min = float(request.form['MIN'])
    fg = float(request.form['FG%'])
    ft = float(request.form['FT%'])
    oreb = float(request.form['OREB'])
    dreb = float(request.form['DREB'])
    ast = float(request.form['AST'])
    stl = float(request.form['STL'])
    blk = float(request.form['BLK'])
    tov = float(request.form['TOV'])
    total_points=gp*pts
    effeciency = pts/min
    reb = dreb + oreb
    di = blk + reb + stl
    ast_tov = ast/tov
    oreb_reb = oreb/reb
    game_impact = (ast+reb)/min
    # Créer un tableau numpy avec les données du joueur
    input_features = np.array([[fg,ft,total_points,effeciency,di,ast_tov, oreb_reb,game_impact]])

    # Standardiser les caractéristiques avant prédiction
    input_features_scaled = scaler.transform(input_features)

    # Faire la prédiction
    prediction = model.predict(input_features_scaled)

    # Résultat : 1 = Carrière > 5 ans, 0 = Carrière <= 5 ans
    result = 'Carrière > 5 ans' if prediction[0] == 1 else 'Carrière <= 5 ans'

    return render_template('index.html', prediction_text=f'Le joueur aura une {result}')


if __name__ == "__main__":
    app.run(debug=True)
