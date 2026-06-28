from abc import ABC, abstractmethod


def _sliding_moves(board, pos, directions):
    moves = []
    r, c = pos
    piece = board.get(pos)
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            target = board.get((nr, nc))
            if target is None:
                moves.append((nr, nc))
            elif target.color != piece.color:
                moves.append((nr, nc))
                break
            else:
                break
            nr += dr
            nc += dc
    return moves


class Piece(ABC):
    def __init__(self, color: str):
        self.color = color  # 'white' or 'black'
        self.has_moved = False

    @property
    @abstractmethod
    def symbol(self) -> str:
        pass

    @property
    @abstractmethod
    def letter(self) -> str:
        pass

    @abstractmethod
    def get_pseudo_moves(self, board, pos: tuple) -> list:
        """Candidate moves ignoring whether they leave own king in check."""
        pass

    def get_moves(self, board, pos: tuple) -> list:
        return [m for m in self.get_pseudo_moves(board, pos)
                if not board.move_leaves_king_in_check(pos, m, self.color)]

    def __repr__(self):
        return f"{self.color[0].upper()}{self.letter}"


class Pawn(Piece):
    @property
    def symbol(self):
        return '♙' if self.color == 'white' else '♟'

    @property
    def letter(self):
        return 'P'

    def get_pseudo_moves(self, board, pos):
        moves = []
        r, c = pos
        direction = -1 if self.color == 'white' else 1
        start_row = 6 if self.color == 'white' else 1

        nr = r + direction
        if 0 <= nr < 8 and board.get((nr, c)) is None:
            moves.append((nr, c))
            if r == start_row:
                nr2 = r + 2 * direction
                if board.get((nr2, c)) is None:
                    moves.append((nr2, c))

        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target = board.get((nr, nc))
                if target is not None and target.color != self.color:
                    moves.append((nr, nc))
                elif (nr, nc) == board.en_passant_target:
                    moves.append((nr, nc))

        return moves


class Rook(Piece):
    @property
    def symbol(self):
        return '♖' if self.color == 'white' else '♜'

    @property
    def letter(self):
        return 'R'

    def get_pseudo_moves(self, board, pos):
        return _sliding_moves(board, pos, [(0, 1), (0, -1), (1, 0), (-1, 0)])


class Knight(Piece):
    @property
    def symbol(self):
        return '♘' if self.color == 'white' else '♞'

    @property
    def letter(self):
        return 'N'

    def get_pseudo_moves(self, board, pos):
        r, c = pos
        piece = board.get(pos)
        moves = []
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target = board.get((nr, nc))
                if target is None or target.color != piece.color:
                    moves.append((nr, nc))
        return moves


class Bishop(Piece):
    @property
    def symbol(self):
        return '♗' if self.color == 'white' else '♝'

    @property
    def letter(self):
        return 'B'

    def get_pseudo_moves(self, board, pos):
        return _sliding_moves(board, pos, [(1, 1), (1, -1), (-1, 1), (-1, -1)])


class Queen(Piece):
    @property
    def symbol(self):
        return '♕' if self.color == 'white' else '♛'

    @property
    def letter(self):
        return 'Q'

    def get_pseudo_moves(self, board, pos):
        return _sliding_moves(board, pos,
                              [(0, 1), (0, -1), (1, 0), (-1, 0),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)])


class King(Piece):
    @property
    def symbol(self):
        return '♔' if self.color == 'white' else '♚'

    @property
    def letter(self):
        return 'K'

    def get_pseudo_moves(self, board, pos):
        """Normal one-step moves only; castling is added in get_moves."""
        r, c = pos
        piece = board.get(pos)
        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    target = board.get((nr, nc))
                    if target is None or target.color != piece.color:
                        moves.append((nr, nc))
        return moves

    def get_moves(self, board, pos):
        moves = [m for m in self.get_pseudo_moves(board, pos)
                 if not board.move_leaves_king_in_check(pos, m, self.color)]

        # Castling — only when not currently in check
        if self.has_moved or board.is_in_check(self.color):
            return moves

        r, c = pos
        back_rank = 7 if self.color == 'white' else 0

        # Kingside
        rook = board.get((back_rank, 7))
        if (isinstance(rook, Rook) and not rook.has_moved
                and board.get((back_rank, 5)) is None
                and board.get((back_rank, 6)) is None
                and not board.square_attacked((back_rank, 5), self.color)
                and not board.square_attacked((back_rank, 6), self.color)):
            moves.append((back_rank, 6))

        # Queenside
        rook = board.get((back_rank, 0))
        if (isinstance(rook, Rook) and not rook.has_moved
                and board.get((back_rank, 1)) is None
                and board.get((back_rank, 2)) is None
                and board.get((back_rank, 3)) is None
                and not board.square_attacked((back_rank, 3), self.color)
                and not board.square_attacked((back_rank, 2), self.color)):
            moves.append((back_rank, 2))

        return moves
