import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from board.game import Game
from board.evaluation import evaluate


def main():
    game = Game()
    game.board.display()
    score = evaluate(game.board)
    print("Board Evaluation:", score)


if __name__ == '__main__':
    main()
