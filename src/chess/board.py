from .piece import Piece, Pawn, Rook, Knight, Bishop, Queen, King


class Board:
    def __init__(self):
        self.grid: list[list[Piece | None]] = [[None] * 8 for _ in range(8)]
        self.en_passant_target: tuple | None = None
        self._setup()

    def _setup(self):
        back_row = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for col, Cls in enumerate(back_row):
            self.grid[0][col] = Cls('black')
            self.grid[1][col] = Pawn('black')
            self.grid[7][col] = Cls('white')
            self.grid[6][col] = Pawn('white')

    # ------------------------------------------------------------------
    # Grid access
    # ------------------------------------------------------------------

    def get(self, pos: tuple) -> Piece | None:
        r, c = pos
        return self.grid[r][c]

    def place(self, pos: tuple, piece: Piece | None):
        r, c = pos
        self.grid[r][c] = piece

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def move(self, from_pos: tuple, to_pos: tuple) -> Piece | None:
        """Execute a legal move. Returns the captured piece (if any)."""
        piece = self.get(from_pos)
        captured = self.get(to_pos)

        # En passant capture
        ep_captured = None
        if isinstance(piece, Pawn) and to_pos == self.en_passant_target:
            ep_pos = (from_pos[0], to_pos[1])
            ep_captured = self.get(ep_pos)
            self.place(ep_pos, None)

        # Update en passant target
        if isinstance(piece, Pawn) and abs(to_pos[0] - from_pos[0]) == 2:
            self.en_passant_target = (
                (from_pos[0] + to_pos[0]) // 2, from_pos[1]
            )
        else:
            self.en_passant_target = None

        # Castling: also move the rook
        if isinstance(piece, King) and abs(to_pos[1] - from_pos[1]) == 2:
            back = from_pos[0]
            if to_pos[1] == 6:  # kingside
                rook = self.get((back, 7))
                self.place((back, 5), rook)
                self.place((back, 7), None)
                rook.has_moved = True
            else:  # queenside
                rook = self.get((back, 0))
                self.place((back, 3), rook)
                self.place((back, 0), None)
                rook.has_moved = True

        self.place(to_pos, piece)
        self.place(from_pos, None)
        piece.has_moved = True

        return captured or ep_captured

    # ------------------------------------------------------------------
    # Check / attack detection
    # ------------------------------------------------------------------

    def find_king(self, color: str) -> tuple | None:
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if isinstance(p, King) and p.color == color:
                    return (r, c)
        return None

    def square_attacked(self, pos: tuple, defender_color: str) -> bool:
        """True if pos is attacked by any piece belonging to the opponent."""
        attacker_color = 'black' if defender_color == 'white' else 'white'
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p is not None and p.color == attacker_color:
                    if pos in p.get_pseudo_moves(self, (r, c)):
                        return True
        return False

    def is_in_check(self, color: str) -> bool:
        king_pos = self.find_king(color)
        return king_pos is not None and self.square_attacked(king_pos, color)

    def move_leaves_king_in_check(
        self, from_pos: tuple, to_pos: tuple, color: str
    ) -> bool:
        """Simulate a move (without touching has_moved) and test for check."""
        fr, fc = from_pos
        tr, tc = to_pos
        piece = self.grid[fr][fc]
        target = self.grid[tr][tc]
        saved_ep = self.en_passant_target

        ep_pos = ep_captured = None
        rook_from = rook_to = rook_piece = None

        # En passant
        if isinstance(piece, Pawn) and to_pos == self.en_passant_target:
            ep_pos = (fr, tc)
            ep_captured = self.grid[fr][tc]
            self.grid[fr][tc] = None

        # Castling: move rook too so sliding attackers are modelled correctly
        if isinstance(piece, King) and abs(tc - fc) == 2:
            back = fr
            if tc == 6:
                rook_from, rook_to = (back, 7), (back, 5)
            else:
                rook_from, rook_to = (back, 0), (back, 3)
            rook_piece = self.grid[rook_from[0]][rook_from[1]]
            self.grid[rook_to[0]][rook_to[1]] = rook_piece
            self.grid[rook_from[0]][rook_from[1]] = None

        self.grid[tr][tc] = piece
        self.grid[fr][fc] = None

        in_check = self.is_in_check(color)

        # Restore everything
        self.grid[fr][fc] = piece
        self.grid[tr][tc] = target
        if ep_pos is not None:
            self.grid[ep_pos[0]][ep_pos[1]] = ep_captured
        if rook_from is not None:
            self.grid[rook_from[0]][rook_from[1]] = rook_piece
            self.grid[rook_to[0]][rook_to[1]] = None
        self.en_passant_target = saved_ep

        return in_check

    # ------------------------------------------------------------------
    # Move enumeration
    # ------------------------------------------------------------------

    def get_all_moves(self, color: str) -> list[tuple]:
        """All legal (from, to) pairs for the given color."""
        moves = []
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p is not None and p.color == color:
                    for to_pos in p.get_moves(self, (r, c)):
                        moves.append(((r, c), to_pos))
        return moves

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self, highlight: set | None = None):
        highlight = highlight or set()
        header = '    a  b  c  d  e  f  g  h'
        divider = '   +--+--+--+--+--+--+--+--+'
        print(header)
        print(divider)
        for r in range(8):
            rank = 8 - r
            row = f' {rank} |'
            for c in range(8):
                p = self.grid[r][c]
                cell = p.symbol if p else ' '
                if (r, c) in highlight:
                    row += f'\033[43m{cell} \033[0m|'
                else:
                    row += f'{cell} |'
            row += f' {rank}'
            print(row)
            print(divider)
        print(header)
