from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import os

app = Flask(__name__, static_folder='static')

# ✅ Load Model & Vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Error: Model files not found. Train the model first.")

loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
tfvect = pickle.load(open(VECTORIZER_PATH, 'rb'))

def fake_news_det(news):
    """Detect whether news is Fake or Real"""
    vectorized_input_data = tfvect.transform([news])
    prediction = loaded_model.predict(vectorized_input_data)[0]
    return "Real" if prediction == 1 else "Fake"

# ✅ Routes

@app.route('/')
def home():
    return render_template('prediction.html')  # Ensure 'prediction.html' exists

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        return redirect(url_for('home'))  # Redirect to home after submission
    return render_template('signin.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact_page():  # Renamed function to avoid duplicate error
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if name and email and message:
            return redirect(url_for('home'))  # Redirect to home after submission

    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"prediction": "Please enter valid news text."}), 400  # Bad Request

        pred = fake_news_det(message)
        return jsonify({"prediction": pred})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal Server Error

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
