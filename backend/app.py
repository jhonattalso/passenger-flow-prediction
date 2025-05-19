from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo treinado
modelo = joblib.load("melhor_modelo.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Dados recebidos em formato JSON
        dados = request.get_json()

        # Espera receber os campos: Estacao, Linha, Horario
        if not all(chave in dados for chave in ['Estacao', 'Linha', 'Horario']):
            return jsonify({'erro': 'JSON deve conter: Estacao, Linha, Horario'}), 400

        # Criar DataFrame com os dados recebidos
        entrada = pd.DataFrame([{
            'Estacao': dados['Estacao'],
            'Linha': dados['Linha'],
            'Horario': dados['Horario']
        }])

        # Fazer a predição
        predicao = modelo.predict(entrada)[0]

        return jsonify({'categoria_prevista': predicao})

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
