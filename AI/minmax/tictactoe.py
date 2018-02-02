"""
Tic Tac Toe game made for learning minmax algorithm.

Tested only on MacOs Sierra.
"""
import sys
import pygame
import numpy as np
from pygame.locals import *

# import deepcopy

## OPTIONS ############################################################################################################

# MINMAX OPTIONS
MAX_DEPTH = 3

# COLORS
WHITE = (250,   250,    250 )
BLACK = (0,     0,      0   )

# GRID
GRID = np.zeros((3, 3), dtype=int)

## MINMAX #############################################################################################################

def heuristic(state):
    """
    """
    pass

def minmax(state, maxplayer, depth):
    """
    """
    # Get all moves from our state.
    # np.argwhere returns arrays of indices.
    moves = np.argwhere(state == 0)
    if depth == MAX_DEPTH or check_state(state) == 1:
        # If we reach the max depth stop and compute heuristic
        return heuristic(state)
    elif maxplayer:
        # Init evaluations to max value
        evaluations = -np.inf
        # Iterate over each moves
        for move in moves:
            new_state = np.copy(state)
            new_state[move[0]][move[1]] = 1 
            heur = minmax(move, False, (depth - 1))
            evaluations = np.maximum(evaluations, heur)
    else:
        evaluations = np.inf
        for move in moves:
            new_state = np.copy(state)
            new_state[move[0]][move[1]] = -1 
            heur = minmax(move, False, (depth - 1))
            evaluations = np.minimum(evaluations, heur)
    return evaluations

def ai_player():
    minmax(np.copy(GRID), True, MAX_DEPTH)
    return X, Y

## TICTACTOE ##########################################################################################################

def init_board(ttt):
    """Initialize the board for Pygame.

    Args:
        ttt: the properly initialized Pygame display variable

    Return:
        board: board initialized with background color and lines
    """
    # Initialize background
    board = pygame.Surface(ttt.get_size())
    board = board.convert()
    board.fill(WHITE)
    # Draw vertical lines
    pygame.draw.line(board, BLACK, (100, 0), (100, 300), 1)
    pygame.draw.line(board, BLACK, (200, 0), (200, 300), 1)
    # Draw horizontal lines
    pygame.draw.line(board, BLACK, (0, 100), (300, 100), 1)
    pygame.draw.line(board, BLACK, (0, 200), (300, 200), 1)
    return board

def check_state():
    """Check if there is a winner.
    """
    # Check horizontal
    sum_h = np.sum(GRID, axis=1)
    # Check vertical
    sum_v = np.sum(GRID, axis=0)
    # Check diagonals
    sum_d1 = GRID[0][0] + GRID[1][1] + GRID[2][2]
    sum_d2 = GRID[2][0] + GRID[1][1] + GRID[0][2]
    # Test if sums corresponds to a winner
    if np.any(sum_h == -3) or np.any(sum_v == -3) or sum_d1 == -3 or sum_d2 == -3:
        # Player 1 won!
        print("player 1 won")
        return 1
    elif np.any(sum_h == 3) or np.any(sum_v == 3) or sum_d1 == 3 or sum_d2 == 3:
        # Player 2 won!
        print("player 2 won")
        return 1
    else:
        # None won
        return 0

def drawstatus(board, player):
    """
    """
    if player is -1:
        message = "Player 1"
    elif player is 1:
        message = "Player 2"
    else:
        message = "Tictactoe!"
    font = pygame.font.Font(None, 24)
    text = font.render(message, 1, (0,0,0))
    board.fill(WHITE, (0, 300, 300, 25))
    board.blit(text,( 10, 300))

def display(ttt, board, player):
    """Display the board.

    Args:
        ttt: is the actual game
        board: is teh Surface background
    """
    drawstatus(board, player)
    ttt.blit(board, (0, 0))
    pygame.display.flip()

def mouse_position(mouseX, mouseY):
    """Determine which cell is clicked by its cell coordinates.

    Args:
        mouseX: mouse position on X
        mouseY: mouse position on Y

    Return:
        (X, Y): position on the board
    """
    if mouseY < 100:
        X = 0
    elif mouseY < 200:
        X = 1
    else:
        X = 2
    if mouseX < 100:
        Y = 0
    elif mouseX < 200:
        Y = 1
    else:
        Y = 2
    return (X, Y)

def draw(board, X, Y, player):
    """Draw a circle or a cross onto the board canvas and modify the grid.
    """
    centerX = (X * 100) + 50
    centerY = (Y * 100) + 50
    if player is -1:
        pygame.draw.circle(board, BLACK, (centerX, centerY), 44, 2)
    else:
        pygame.draw.line(board, BLACK, (centerX - 33, centerY - 33),
                        (centerX + 33, centerY + 33), 2)
        pygame.draw.line(board, BLACK, (centerX + 33, centerY - 33),
                        (centerX - 33, centerY + 33), 2)

def play(board, player):
    """Analyze the player click, display it and play.

    Args:
        board:
        player:
    """
    (mouseY, mouseX) = pygame.mouse.get_pos()
    (X, Y) = mouse_position(mouseX, mouseY)
    if GRID[X][Y] != 0:
        return
    draw(board, X, Y, player)
    return X, Y

def main():
    """The game loop function."""
    # Initialize some Pygame
    pygame.init()
    ttt = pygame.display.set_mode((300, 325))
    pygame.display.set_caption = ('Tic Tac Toe')
    board = init_board(ttt)

    # Initialize some game variables
    loop = True
    player = -1 # Choose first player (-1 or 1)

    # Loop game
    while (loop):
        for event in pygame.event.get():
            if event.type == QUIT:
                loop = False
            elif event.type == MOUSEBUTTONDOWN and player == -1:
                X, Y = play(board, player)
            else:
                X, Y = ai_player()
            GRID[X][Y] = player            
            check_state()
            display(ttt, board, player)
            player = (-1) * player


if __name__ == '__main__':
    main()
