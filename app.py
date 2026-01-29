from flask import Flask, render_template, request, jsonify
import serpapi

app = Flask(__name__, template_folder='templates',static_folder='templates')

API_KEY = "fd4a66c6dce02ccaf5a0d8b8d65d48d89f21883f50826fb2d18369f910213b75"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    
    params = {
        "engine": "google_flights",
        "departure_id": data.get('departure_airport'),
        "arrival_id": data.get('arrival_airport'),
        "outbound_date": data.get('departure_date'),
        "return_date": data.get('return_date'),
        "currency": "EUR",
        "hl": "en",
        "api_key": API_KEY,
    }
    
    try:
        search = serpapi.search(params)
        results = search.as_dict()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/search-return', methods=['POST'])
def search_return():
    data = request.json
    
    # Garder les mêmes paramètres que la recherche initiale + ajouter le departure_token
    params = {
        "engine": "google_flights",
        "departure_id": data.get('departure_airport'),  # Garder le même sens
        "arrival_id": data.get('arrival_airport'),      # Garder le même sens
        "outbound_date": data.get('departure_date'),    # Date de départ originale
        "return_date": data.get('return_date'),         # Date de retour
        "departure_token": data.get('departure_token'), # Token du vol sélectionné
        "currency": "EUR",
        "hl": "en",
        "api_key": API_KEY,
    }
    
    try:
        print(f"Paramètres envoyés à l'API: {params}")  # Debug
        search = serpapi.search(params)
        results = search.as_dict()
        print(f"Résultats reçus: {results}")  # Debug
        return jsonify(results)
    except Exception as e:
        print(f"Erreur détaillée: {e}")  # Debug
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/book', methods=['POST'])
def book():
    data = request.json
    booking_token = data.get('booking_token')
    
    if not booking_token:
        return jsonify({"error": "Booking token manquant"}), 400
    
    params = {
        "engine": "google_flights",
        "booking_token": booking_token,
        "api_key": API_KEY,
    }
    
    try:
        print(f"Paramètres de booking: {params}")  # Debug
        search = serpapi.search(params)
        results = search.as_dict()
        print(f"Résultats de booking: {results}")  # Debug
        return jsonify(results)
    except Exception as e:
        print(f"Erreur de booking: {e}")  # Debug
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
