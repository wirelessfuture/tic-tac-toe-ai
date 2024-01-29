import itertools
import numpy as np
import csv


def is_win(board, player):
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True
        if all(board[j, i] == player for j in range(3)):
            return True
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2 - i] == player for i in range(3)):
        return True
    return False


def get_optimal_move(board):
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = 1
                if is_win(board, 1):
                    return board
                board[i, j] = 0
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = 1
                return board
    return board


def generate_all_game_states():
    all_states = itertools.product([-1, 0, 1], repeat=9)
    with open("training_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["game_state", "optimal_move"])
        for state in all_states:
            board = np.array(state).reshape(3, 3)
            optimal_move = get_optimal_move(np.copy(board))
            writer.writerow([board.tolist(), optimal_move.tolist()])


generate_all_game_states()
