// -------------------------------------------------------------------------- //
//          mouvements.py - Gestion des mouvements des pieces de l'echiquier  //
// ainsi que de la communication avec main.py et interface-web.html           //
// -------------------------------------------------------------------------- //
// Auteur : Corentin SALVI                                                    //
// Date : 2024-10-13                                                          //
// -------------------------------------------------------------------------- //

// -------------------------------------------
// Fonction qui permet d'avoir la postion 3D d'une case à partir de sa notation échiquéenne
// -------------------------------------------
//#region
function getPositionFromNotation(notation) {
    const files = { 'a': -8.8, 'b': -6.3, 'c': -3.8, 'd': -1.3, 'e': 1.3, 'f': 3.8, 'g': 6.3, 'h': 8.8 };
    const ranks = { '8': -8.8, '7': -6.3, '6': -3.8, '5': -1.3, '4': 1.3, '3': 3.8, '2': 6.3, '1': 8.8 };
    const file = notation[0].toLowerCase();
    const rank = notation[1];
    return { x: files[file], z: ranks[rank], y: 8.8 };
}
//#endregion

// -------------------------------------------
// Fonction qui permet de trouver une pièce à une position donnée
// -------------------------------------------
//#region
function findPieceAtPosition(notation, scene) {
    const targetPos = getPositionFromNotation(notation);
    let foundPiece = null;
    const epsilon = 0.5;
    let found=false;
    window.Chess3D.scene.traverse((child) => {
        if (found) return;
        if (child.userData.isPiece) {
            if (
                Math.abs(child.position.x - targetPos.x) < epsilon &&
                Math.abs(child.position.z - targetPos.z) < epsilon
            ) {
                foundPiece = child;
                found=true;
            }
        }
    });
    return foundPiece;
}
//#endregion

// -------------------------------------------
// Fonction qui permet de capturer une pièce (la retirer de la scène)
// -------------------------------------------
//#region
function capturePiece(piece) {
    if (piece && piece.parent) {
        piece.parent.remove(piece);
    }
}
//#endregion

// -------------------------------------------
// Fonction qui permet de déplacer une pièce à une nouvelle position
// -------------------------------------------
//#region
function movePiece(piece, notation, animate = true) {
    const newPos = getPositionFromNotation(notation);
    piece.position.x = newPos.x;
    piece.position.z = newPos.z;
    piece.userData.notation = notation;
}
//#endregion

// -------------------------------------------
// Fonctions qui recupere le mouvement saisi dans l'input et l'envoie au backend
// -------------------------------------------
//#region
function sendMoveFromInput() {
    const move = document.getElementById("moveInput").value;
    document.getElementById('moveInput').value = '';
    console.log("Coup saisi :", move); 
    sendMove(move);
}
//#endregion

// -------------------------------------------
// Fonction qui envoie le mouvement au backend et traite la réponse
// -------------------------------------------
//#region
function sendMove(move) {
    fetch("/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ move: move })
    })
    .then(res => res.json())
    .then(data => {
        console.log("Réponse Python:", data);
        if (data.user_move.length === 4) {
        const from = data.user_move.slice(0, 2); 
        const to = data.user_move.slice(2);
        if (data.user_move=="e1g1" || data.user_move=="e1c1") {
            if (to === "g1") {
                const king = findPieceAtPosition("e1", window.Chess3D.scene);
                const rook = findPieceAtPosition("h1", window.Chess3D.scene);
                movePiece(king, "g1");
                movePiece(rook, "f1");
            } else if (to === "c1") { 
                const king = findPieceAtPosition("e1", window.Chess3D.scene);
                const rook = findPieceAtPosition("a1", window.Chess3D.scene);
                movePiece(rook, "d1");
                movePiece(king, "c1");
            }
            data.user_castled=true;
        }
        else{
        const capturedPiece = findPieceAtPosition(to, window.Chess3D.scene);
        if (capturedPiece) {
            capturePiece(capturedPiece);
        }     
        const piece = findPieceAtPosition(from, window.Chess3D.scene);
        if (piece) {
            movePiece(piece, to);
        }
        }
    }
        if (data.bot_move.length === 4) {
            const from = data.bot_move.slice(0, 2); 
            const to = data.bot_move.slice(2);     
            if (data.bot_move=="e8g8" || data.bot_move=="e8c8") {
                if (to === "g8") {
                    const king = findPieceAtPosition("e8", window.Chess3D.scene);
                    const rook = findPieceAtPosition("h8", window.Chess3D.scene);
                    movePiece(king, "g8");
                    movePiece(rook, "f8");
                } else if (to === "c8") { 
                    const king = findPieceAtPosition("e8", window.Chess3D.scene);
                    const rook = findPieceAtPosition("a8", window.Chess3D.scene);   
                    movePiece(king, "c8");
                    movePiece(rook, "d8");
                }
                data.bot_castled=true;
            }
            else{
            const capturedPiece = findPieceAtPosition(to, window.Chess3D.scene);
            if (capturedPiece) {
                capturePiece(capturedPiece);
            }
            const piece = findPieceAtPosition(from, window.Chess3D.scene);
            if (piece) {
                movePiece(piece, to);
            }
        }
            
        }
    })
    .catch(err => console.error(err));
}
//#endregion

// -------------------------------------------