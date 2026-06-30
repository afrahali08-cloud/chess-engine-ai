from .piece import Pawn, Knight, Bishop, Rook, Queen, King

PIECE_VALUES = {
  Pawn : 1,
  Knight : 3,
  Bishop : 3, 
  Rook : 5,
  Queen : 9,
  King : 0
}

def evaluate(board):
  score = 0
  for row in board.grid:
    for piece in row:
      if piece is None:
        continue

value = PIECE_VALUES[type(piece)]
if piece.color == "white":
  score += value
else:
  score -= value

return score
  

        
