# -------------------------------------------------------------------------- #
#          main.py - Application web d'échecs avec Flask et Stockfish        #
# -------------------------------------------------------------------------- #
# Auteur : Corentin SALVI                                                    #
# Date : 2024-10-13                                                          #
# -------------------------------------------------------------------------- #

# -------------------------------------------
# Importation des bilbiothèques 
# -------------------------------------------
#region
from flask import Flask, render_template, request, jsonify, session
import chess
import chess.engine
#endregion

# -------------------------------------------
# Initialisation de l'application Flask 
# -------------------------------------------
#region
app = Flask(__name__)
app.secret_key = "supersecretkey"  
#endregion

# -------------------------------------------
# Initialisation du moteur d'échecs et de la difficulté
# -------------------------------------------
#region
engine = chess.engine.SimpleEngine.popen_uci("Executable/stockfish/stockfish-windows-x86-64-avx2.exe")
DIFFICULTY = 1 # profondeur par défaut
#endregion

# -------------------------------------------
# Obtention ou création du plateau de jeu
# -------------------------------------------
#region
def get_board():
    if "fen" in session:
        board = chess.Board(session["fen"])
    else:
        board = chess.Board()
        session["fen"] = board.fen()
    return board
#endregion


# -------------------------------------------
# Obtention du statut du je
# -------------------------------------------
#region
def get_status(board):
    if board.is_checkmate():
        return "Échec et mat !"
    elif board.is_stalemate():
        return "Pat (égalité) !"
    elif board.is_insufficient_material():
        return "Matériel insuffisant pour gagner (égalité) !"
    elif board.is_fivefold_repetition():
        return "Répétition 3 fois de la même position (égalité) !"
    else:
        return "Partie en cours..."
#endregion


# -------------------------------------------
# Route flask et connexion avec le frontend
# -------------------------------------------
#region
@app.route("/")
def index():
    session.clear()
    return render_template("interface-web.html")
#endregion 

# -------------------------------------------
# Route pour gérer les coups
# -------------------------------------------
@app.route("/move", methods=["POST"])
def move():
    # Récupération des données JSON et du plateau
    data = request.json
    user_move = data.get("move")
    difficulty = data.get("difficulty", DIFFICULTY)
    board = get_board()
    # Coup de l'utilisateur
    try:
        # Validation du coup
        move = chess.Move.from_uci(user_move)
        if move not in board.legal_moves:
            return jsonify({"error": "Coup illégal"}), 400
        # region Capture avant le coup utilisateur
        captured_piece = None
        if board.is_capture(move):
            captured_square = move.to_square
            captured_piece_obj = board.piece_at(captured_square)
            if captured_piece_obj:
                captured_piece = captured_piece_obj.symbol()
        # endregion
        # Effectuer le coup
        board.push(move)
        session["fen"] = board.fen()
        user_move_uci = move.uci()
        # region Vérifier si l'utilisateur a roqué
        roque=False
        if board.is_castling(move):
            roque=True
        # endregion
        # region Vérifier si la partie est terminée après le coup
        game_over = board.is_game_over()
        if (game_over):
            result = board.result()
        else:
            result = None
        status = get_status(board)
        # endregion
        # Coup du robot d'échecs
        bot_move_uci = None
        bot_captured_piece = None
        if not game_over:
            # Choix du coup par le robot d'échecs
            bot_result = engine.play(board, chess.engine.Limit(depth=int(difficulty)))
            bot_move = bot_result.move
            # region Capture avant le coup du robot
            if board.is_capture(bot_move):
                bot_captured_square = bot_move.to_square
                bot_captured_piece_obj = board.piece_at(bot_captured_square)
                if bot_captured_piece_obj:
                    bot_captured_piece = bot_captured_piece_obj.symbol()
            # endregion
            # Effectuer le coup du robot
            board.push(bot_move)
            session["fen"] = board.fen()
            bot_move_uci = bot_move.uci()
            status = get_status(board)
            # region Vérifier si le robot a roqué
            bot_roque=False
            if board.is_castling(bot_move):
                bot_roque=True
            # endregion
            # region Vérifier si la partie est terminée après le coup du robot
            game_over = board.is_game_over()
            if (game_over):
                result = board.result()
            else:
                result = None
            status = get_status(board)
            # endregion
        # Retour des informations au frontend
        return jsonify({
            "fen": board.fen(),
            "game_over": game_over,
            "result": result,
            "status": status,
            "user_move": user_move_uci,
            "bot_move": bot_move_uci,
            "user_captured_piece": captured_piece,
            "bot_captured_piece": bot_captured_piece,
            "user_castled": roque,
            "bot_castled": bot_roque
        })
    # Gestion des erreurs
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)