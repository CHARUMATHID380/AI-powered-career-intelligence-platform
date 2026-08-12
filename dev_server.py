import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
# Now import your function handlers
from api.predict import handler as predict_handler
from api.predict_narrow import handler as predict_narrow_handler

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    return predict_handler(request)

@app.route('/api/predict_narrow', methods=['POST'])
def predict_narrow():
    return predict_narrow_handler(request)

if __name__ == '__main__':
    app.run(port=5000, debug=True)