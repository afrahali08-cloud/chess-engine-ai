import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chess.game import Game


def main():
    game = Game()
    game.play()


if __name__ == '__main__':
    main()
