from .board import Board
from .piece import Pawn, Rook, Knight, Bishop, Queen, King


PROMOTIONS = {'q': Queen, 'r': Rook, 'b': Bishop, 'n': Knight}


class Game:
    def __init__(self):
        self.board = Board()
        self.current_turn = 'white'
        self.move_history: list[tuple] = []

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_move(text: str) -> tuple[tuple | None, tuple | None]:
        """Parse 'e2e4' (or 'e2 e4') into ((fr,fc), (tr,tc))."""
        s = text.strip().replace(' ', '').lower()
        if len(s) < 4:
            return None, None
        try:
            fc = ord(s[0]) - ord('a')
            fr = 8 - int(s[1])
            tc = ord(s[2]) - ord('a')
            tr = 8 - int(s[3])
        except (ValueError, IndexError):
            return None, None
        if not all(0 <= x < 8 for x in (fr, fc, tr, tc)):
            return None, None
        return (fr, fc), (tr, tc)

    @staticmethod
    def pos_to_alg(pos: tuple) -> str:
        r, c = pos
        return f"{chr(ord('a') + c)}{8 - r}"

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def play(self):
        print()
        print('Chess — enter moves as e2e4 (from→to). "help" for commands.')
        print()

        while True:
            self.board.display()
            print()

            legal = self.board.get_all_moves(self.current_turn)
            in_check = self.board.is_in_check(self.current_turn)

            if not legal:
                if in_check:
                    loser = self.current_turn
                    winner = 'black' if loser == 'white' else 'white'
                    print(f'Checkmate! {winner.capitalize()} wins!')
                else:
                    print("Stalemate — it's a draw.")
                break

            if in_check:
                print(f'  ** {self.current_turn.capitalize()} is in check! **')

            print(f'{self.current_turn.capitalize()}> ', end='', flush=True)
            raw = input().strip()

            if raw.lower() in ('quit', 'exit', 'q'):
                print('Game ended.')
                break

            if raw.lower() == 'help':
                self._print_help()
                continue

            if raw.lower().startswith('moves '):
                self._show_moves(raw[6:].strip(), legal)
                continue

            from_pos, to_pos = self.parse_move(raw)
            if from_pos is None:
                print("  Bad format — try 'e2e4'.")
                continue

            piece = self.board.get(from_pos)
            if piece is None:
                print('  No piece at that square.')
                continue
            if piece.color != self.current_turn:
                print("  That piece belongs to the opponent.")
                continue
            if (from_pos, to_pos) not in legal:
                print('  Illegal move.')
                continue

            self.board.move(from_pos, to_pos)
            self.move_history.append((from_pos, to_pos))

            # Pawn promotion
            moved = self.board.get(to_pos)
            if isinstance(moved, Pawn):
                back = 0 if self.current_turn == 'white' else 7
                if to_pos[0] == back:
                    self._handle_promotion(to_pos)

            self.current_turn = 'black' if self.current_turn == 'white' else 'white'
            print()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_promotion(self, pos: tuple):
        color = self.board.get(pos).color
        while True:
            choice = input('  Promote to (q/r/b/n): ').strip().lower()
            if choice in PROMOTIONS:
                self.board.place(pos, PROMOTIONS[choice](color))
                break
            print('  Choose q, r, b, or n.')

    def _show_moves(self, square: str, legal: list):
        if len(square) != 2:
            print("  Usage: moves e2")
            return
        try:
            c = ord(square[0]) - ord('a')
            r = 8 - int(square[1])
        except (ValueError, IndexError):
            print("  Bad square.")
            return
        targets = {to for (frm, to) in legal if frm == (r, c)}
        if not targets:
            print(f'  No legal moves from {square}.')
        else:
            alg = sorted(self.pos_to_alg(t) for t in targets)
            print(f'  Legal moves: {", ".join(alg)}')
        self.board.display(highlight=targets)

    @staticmethod
    def _print_help():
        print('  Commands:')
        print('    e2e4        — move piece from e2 to e4')
        print('    moves e2    — show legal moves for piece on e2')
        print('    quit        — end the game')
