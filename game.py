import pygame
import numpy as np
import time
from keras.models import load_model

pygame.init()

WIDTH, HEIGHT = 300, 300
LINE_WIDTH = 10
BOARD_ROWS = 3
BOARD_COLS = 3
CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 15
CROSS_WIDTH = 60
SPACE = 20

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (255, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe AI")
screen.fill(BG_COLOR)


def draw_lines():
    pygame.draw.line(screen, LINE_COLOR, (0, 100), (300, 100), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 200), (300, 200), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (100, 0), (100, 300), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (200, 0), (200, 300), LINE_WIDTH)


def draw_figures(board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 1:
                pygame.draw.circle(
                    screen,
                    CIRCLE_COLOR,
                    (int(col * 100 + 100 / 2), int(row * 100 + 100 / 2)),
                    CIRCLE_RADIUS,
                    CIRCLE_WIDTH,
                )
            elif board[row][col] == -1:
                pygame.draw.line(
                    screen,
                    CROSS_COLOR,
                    (col * 100 + SPACE, row * 100 + SPACE),
                    (col * 100 + 100 - SPACE, row * 100 + 100 - SPACE),
                    CROSS_WIDTH,
                )

                pygame.draw.line(
                    screen,
                    CROSS_COLOR,
                    (col * 100 + 100 - SPACE, row * 100 + SPACE),
                    (col * 100 + SPACE, row * 100 + 100 - SPACE),
                    CROSS_WIDTH,
                )


def ai_move(board, model):
    flat_board = board.flatten()
    flat_board = flat_board.reshape((1, 9))

    prediction = model.predict(flat_board)[0]
    best_move = np.argmax(prediction)

    row, col = divmod(best_move, 3)
    if board[row][col] == 0:
        board[row][col] = -1


def check_end(board):
    for i in range(3):
        if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
            return True
        if np.all(board[i, :] == -1) or np.all(board[:, i] == -1):
            return True

    if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
        return True
    if np.all(np.diag(board) == -1) or np.all(np.diag(np.fliplr(board)) == -1):
        return True

    if not np.any(board == 0):
        return True

    return False


def main():
    board = np.zeros((BOARD_ROWS, BOARD_COLS))
    game_over = False
    model = load_model("game_ai_model.keras")

    draw_lines()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouseX = event.pos[0]  # x
                mouseY = event.pos[1]  # y

                clicked_row = int(mouseY // 100)
                clicked_col = int(mouseX // 100)

                if board[clicked_row][clicked_col] == 0:
                    board[clicked_row][clicked_col] = 1
                    draw_figures(board)
                    pygame.display.update()

                    if check_end(board):
                        game_over = True
                    else:
                        ai_move(board, model)
                        draw_figures(board)
                        pygame.display.update()

                        if check_end(board):
                            game_over = True

            if game_over:
                time.sleep(2)
                board = np.zeros((BOARD_ROWS, BOARD_COLS))
                game_over = False
                screen.fill(BG_COLOR)
                draw_lines()
                pygame.display.update()

        pygame.display.update()


main()
