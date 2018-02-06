"""
Tic Tac Toe game made for learning minmax algorithm.

To play this game you need Pygame and Numpy. Check the `requirements.py` for more informations.
"""
import sys
import pygame
import argparse
import numpy as np
from pygame.locals import *

## OPTIONS ############################################################################################################

# COLORS
WHITE = (250,   250,    250 )
BLACK = (0,     0,      0   )

# GRID
GRID = np.zeros((3, 3), dtype=int)

## MINMAX #############################################################################################################

def heuristic(state, depth, player):
    """Compute the score for minmax algorithm."""
    winner = check_state(state)
    if winner == player:
        return 10 - depth
    elif winner == -player:
        return depth - 10
    else:
        return 0
    

def minmax(state, maxplayer, depth, player):
    """Minmax algorithm.
    
    Args:
        state: numpy array (3, 3) representing the current state
        maxplayer: bool corresponding to max player turn (True) or not (False)
        depth: the current depth, it increases till it reach MAX_DEPTH
        player: -1 or 1, depending on which started

    Return:
        evals: the score for one state given
    """
    if check_state(state) != 0 or depth == MAX_DEPTH:
        return heuristic(state, depth, player)

    coordX, coordY = np.where(state == 0)

    if maxplayer:
        evals = -np.inf
        for X, Y in zip(coordX, coordY):
            new_state = np.copy(state)
            new_state[X, Y] = player
            heur = minmax(new_state, False, (depth + 1), player)
            evals = np.maximum(evals, heur)
    else:
        evals = np.inf
        for X, Y in zip(coordX, coordY):
            new_state = np.copy(state)
            new_state[X, Y] = -player
            heur = minmax(new_state, True, (depth + 1), player)
            evals = np.minimum(evals, heur)

    return evals

def get_best_move(player):
    """Return the best move using the minmax algorithm."""
    global GRID, MAX_DEPTH

    best_eval = -np.inf
    best_move = list()

    coordX, coordY = np.where(GRID == 0)

    for X, Y in zip(coordX, coordY):
        new_state = np.copy(GRID)
        new_state[X, Y] = player
        evals = minmax(new_state, False, 0, player)

        if evals > best_eval:
            best_eval = evals
            best_move = [X, Y]

    return best_move

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

def check_state(state):
    """Check if there is a winner."""
    # Check horizontal
    sum_h = np.sum(state, axis=1)
    # Check vertical
    sum_v = np.sum(state, axis=0)
    # Check diagonals
    sum_d1 = state[0, 0] + state[1, 1] + state[2, 2]
    sum_d2 = state[2, 0] + state[1, 1] + state[0, 2]
    # Test if sums corresponds to a winner
    if np.any(sum_h == -3) or np.any(sum_v == -3) or sum_d1 == -3 or sum_d2 == -3:
        # Player 1 won!
        return -1
    elif np.any(sum_h == 3) or np.any(sum_v == 3) or sum_d1 == 3 or sum_d2 == 3:
        # Player 2 won!
        return 1
    elif np.any(state == 0):
        # Still
        return 0
    else:
        # Draw
        return 10

def drawstatus(board, player):
    """Display a message to indicate which player is currently playing."""
    if player is -1:
        message = "Player 1"
    else:
        message = "Player 2"
    font = pygame.font.Font(None, 24)
    text = font.render(message, 1, (0,0,0))
    board.fill(WHITE, (0, 300, 300, 25))
    board.blit(text,( 10, 300))

def draw(board, X, Y, player):
    """Draw a circle or a cross onto the board canvas and modify the grid."""
    centerY = (X * 100) + 50
    centerX = (Y * 100) + 50
    if player is -1:
        pygame.draw.circle(board, BLACK, (centerX, centerY), 44, 2)
    else:
        pygame.draw.line(board, BLACK, (centerX - 33, centerY - 33),
                        (centerX + 33, centerY + 33), 2)
        pygame.draw.line(board, BLACK, (centerX + 33, centerY - 33),
                        (centerX - 33, centerY + 33), 2)

def display(ttt, board, player, X, Y):
    """Display the game.

    Call draw if we have set X and Y. Otherwise just draw the background and the status.

    Args:
        ttt: is the actual game
        board: is the Surface background
    """
    if X is not None or Y is not None:
        draw(board, X, Y, player)
    drawstatus(board, player)
    ttt.blit(board, (0, 0))
    pygame.display.flip()

def mouse_position(mouseX, mouseY):
    """Transform mouses coordinates into array indexes.

    Args:
        mouseX: mouse position on X, from 0 to 300
        mouseY: mouse position on Y, from 0 to 300

    Return:
        X, Y: position on the board. Each one is either 0, 1 or 2.
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
    return X, Y

def play(board, player):
    """Get human player turn.

    Args:
        board: the Surface background
        player: should be -1 (Player 1) or 1 (Player 2)

    Returns:
        X, Y: array indexes for the grid array
    """
    mouseX, mouseY = pygame.mouse.get_pos()
    X, Y = mouse_position(mouseX, mouseY)
    return X, Y

def check_winner():
    global GRID
    winner = check_state(GRID)
    if winner != 0:
        if winner == 1:
            print("Player 1 won!")
        elif winner == 2:
            print("Player 2 won!")
        elif winner == -1:
            print("Draw game")
        sys.exit()


def main(args):
    """The game loop function."""
    global GRID

    # Initialize some Pygame
    pygame.init()
    ttt = pygame.display.set_mode((300, 325))
    pygame.display.set_caption = ('Tic Tac Toe')
    board = init_board(ttt)

    # Initialize some game variables
    loop = True
    if args['startscd']:
        player = 1
    else:
        player = -1
    X, Y = None, None

    # Loop game
    while (loop):
        display(ttt, board, player, X, Y)
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            elif player == -1:
                if event.type != MOUSEBUTTONDOWN: continue
                X, Y = play(board, player)
            elif player == 1:
                X, Y = get_best_move(player)
            GRID[X, Y] = player
            player = (-1) * player
            check_winner()

    # Quit
    return

## MAIN ###############################################################################################################

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="This is a Tic Tac Toe game made for learning minmax algorithm. Enjoy!")
    parser.add_argument("--maxdepth", type=int, default=100, help="Define the max depth for minmax algorithm. The more the better the AI is. Set it very low (1 or 2 for example) for a dumb AI.")
    parser.add_argument("--startscd", default=False, action='store_true', help="The human player start second.")
    args = vars(parser.parse_args())
    
    # Set the global max depth
    MAX_DEPTH = args['maxdepth']

    # Call the main game
    main(args)
