# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import chess
import numpy as np

app = Flask(__name__, template_folder='.') # Tell Flask to look for templates in the same directory

# Load the trained model
try:
    # Corrected: Added quotes around the filename
    model = joblib.load('chess_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'chess_model.pkl' not found. Please run the training script first.")
    model = None

# A function to extract numerical features from a FEN string
def fen_to_features(fen_string):
    """
    Converts a FEN string into a numerical vector representing the board state.
    
    Args:
        fen_string (str): The FEN string of the chess position.
    
    Returns:
        list: A list of numerical features representing the board and whose turn it is.
    """
    try:
        board = chess.Board(fen_string)
    except ValueError:
        return None

    # Map chess pieces to numerical values
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
    }

    # Create a vector representing the 64 squares of the board
    board_vector = np.zeros(64, dtype=int)
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece is not None:
            board_vector[i] = piece_map.get(piece.symbol(), 0)

    features = list(board_vector)
    features.append(1 if board.turn == chess.WHITE else 0)
    return features


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predicting the outcome of a chess game from a FEN string.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model and save it as chess_model.pkl."}), 500
    
    data = request.get_json(force=True)
    fen_string = data.get('fen')

    # Corrected: Check for an empty FEN string
    if not fen_string:
        return jsonify({"error": "No FEN string provided."}), 400

    # Corrected: Call the fen_to_features function to get the features
    features = fen_to_features(fen_string)
    if features is None:
        return jsonify({"error": "Invalid FEN string."}), 400

    features_reshaped = np.array(features).reshape(1, -1)
    prediction = model.predict(features_reshaped)
    
    # Corrected: The prediction values are the strings from the label column, e.g. "White Win"
    if prediction[0] == 'White Win':
        result = "White is predicted to win."
    elif prediction[0] == 'Black Win':
        result = "Black is predicted to win."
    else:
        result = "The game is predicted to be a draw."

    return jsonify({"prediction": result})


@app.route('/')
def index():
    """
    Serves the HTML frontend.
    """
    return render_template('./templates/index.html')


if __name__ == '__main__':
    app.run(debug=True)
