from . import *
from brumaire.board import BOARD_VEC_SIZE

TURN = 10

class Recorder:
    first_boards: NDFloatArray
    """
    shape: `(5, board_num, BOARD_VEC_SIZE)`
    """

    declarations: NDIntArray
    """
    shape: `(5, board_num, 4)`
    """

    boards: NDFloatArray
    """
    shape: `(5, board_num, TURN, BOARD_VEC_SIZE)`
    """

    decisions: NDIntArray
    """
    shape: `(5, board_num, TURN, 54)`
    """

    rewards: NDFloatArray
    """
    shape: `(5, board_num, TURN)`
    """

    winners: NDIntArray
    """
    shape: `(5, board_num)`
    """

    def __init__(self, board_num: int) -> None:
        self.first_boards = np.zeros((5, board_num, BOARD_VEC_SIZE))
        self.declarations = np.zeros((5, board_num, 4))

        self.boards = np.zeros((5, board_num, TURN, BOARD_VEC_SIZE))
        self.decisions = np.zeros((5, board_num, TURN, 54))
        self.rewards = np.zeros((5, board_num, TURN))

        self.winners = np.zeros((5, board_num))
