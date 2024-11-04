'''
    COMP 4010 [Fall 2024]
    Carleton University

    Project Details:
        ~ RL Chess Agent ~
        Date: October 15, 2024

        Group Members:
            Kyle Eng 101192595
'''

import numpy as np
from scipy.linalg import block_diag
import matplotlib
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
import random

EMPTY_SQUARE = 0
KING = 1
QUEEN = 2
ROOK = 3
BISHOP = 4
KNIGHT = 5
PAWN = 6

WHITE = 'WHITE'
BLACK = 'BLACK'

CONVERT_PAWN_TO_QUEEN_REWARD = 10
PAWN_VALUE = 1
KNIGHT_VALUE = 3
BISHOP_VALUE = 3
ROOK_VALUE = 5
QUEEN_VALUE = 10
WIN_REWARD = 100
LOSS_REWARD = -100

INVALID_ACTION_REWARD = -10
VALID_ACTION_REWARD = 10

class Piece:
    icon: str
    type: int
    color: str
    value: float

PIECES = [
    Piece(icon="♙", color=BLACK, type=PAWN,   value=PAWN_VALUE),
    Piece(icon="♘", color=BLACK, type=KNIGHT, value=KNIGHT_VALUE),
    Piece(icon="♗", color=BLACK, type=BISHOP, value=BISHOP_VALUE),
    Piece(icon="♖", color=BLACK, type=ROOK,   value=ROOK_VALUE),
    Piece(icon="♕", color=BLACK, type=QUEEN,  value=QUEEN_VALUE),
    Piece(icon="♔", color=BLACK, type=KING,   value=0),
    Piece(icon=".",  color=None,  type=EMPTY_SQUARE, value=0),
    Piece(icon="♚", color=WHITE, type=KING,   value=0),
    Piece(icon="♛", color=WHITE, type=QUEEN,  value=QUEEN_VALUE),
    Piece(icon="♜", color=WHITE, type=ROOK,   value=ROOK_VALUE),
    Piece(icon="♝", color=WHITE, type=BISHOP, value=BISHOP_VALUE),
    Piece(icon="♞", color=WHITE, type=KNIGHT, value=KNIGHT_VALUE),
    Piece(icon="♟", color=WHITE, type=PAWN,   value=PAWN_VALUE),
]

'''
    Black = (-)
    White = (+)
'''
board_LosAlamos = [
                    [-3, -5, -2, -1, -5, -3],
                    [-6, -6, -6, -6, -6, -6],
                    [0] * 6,
                    [0] * 6,
                    [6, 6, 6, 6, 6, 6],
                    [3, 5, 2, 1, 5, 3],
                ]

# OPPONENT POLICY
def make_random_policy(np_random, bot_player):
    def random_policy(env):
        # moves = env.get_possible_moves(player=bot_player)
        moves = env.possible_moves
        # No moves left
        if len(moves) == 0:
            return "resign"
        else:
            idx = np.random.choice(np.arange(len(moves)))
            return moves[idx]

    return random_policy

class Chess(gym.Env):
    
    def __init__(self, render_mode=None): 

        # Setup Board/State
        self._size = 6
        self.layout = board_LosAlamos
        self.turn = WHITE
        self.done = False

        '''
            Setup PLayers (RL Agent vs Opponent(Randomn Moves))
            Randomize whether agent is White/Black
        '''
        self.player_colour = random.choice([WHITE, BLACK])
        if (self.player_colour == WHITE):
            self.opponent_colour = BLACK
        else:
            self.opponent_colour = WHITE
        
        # Black vs White Pieces, 6x6 = size of board 
        self.observation_space = gym.spaces.Box(-6, 6, (6, 6))
        # + 1 for White vs Black Turn
        self.action_space = gym.spaces.Discrete(36 * 36 + 1)

        self.n_states = np.prod(self.observation_space.shape)
        
        '''
        6 pawns 
            *NO PROMOTION
            1 Forward
            Diagonal Left or Right
            6pieces * 3moves = 18

        2 rooks
            Can move:
                - horizontally to 5 different cells in the row
                - horizontally to 5 different cells in the column
            2pieces * 10moves = 20

        2 knights
            Can move: in 8 ways
            2pieces * 8moves = 16

        1 Queen
            If in the middle can move to 19 different positions
            1 piece x 19moves = 19

        1 King
            If in the middle can move to 8 different positions
            1 piece x 8moves = 8

        18 + 20 + 16 + 19 + 8 = 81
        
        In every state, there are 81 actions
        '''
        self.n_actions = 81
    
    def reset(self):
        # Setup Board/State
        self.layout = board_LosAlamos
        self.turn = WHITE
        self.done = False

        '''
            Setup PLayers (RL Agent vs Opponent(Randomn Moves))
            Randomize whether agent is White/Black
        '''
        self.player_colour = random.choice([WHITE, BLACK])
        if (self.player_colour == WHITE):
            self.opponent_colour = BLACK
        else:
            self.opponent_colour = WHITE

        state = (self.layout, self.turn)

        return state

    def step(self, action):
        '''
            Format:
                State = (board, turn)
                action = (start_position, end position)
        '''
        # Move Chess Piece
         # TODO: implement movement 
        self.move_piece(action)

        state = (self.board, self.turn)
        reward = self.calculate_reward(state)
        terminated = self.check_winner()
 
        return state, reward, terminated, False, {}

    def get_possible_actions(self):
        '''
            TODO: Returns a list [] of actions with format
            [(start, end), (start, end), (start, end), ...]

            For all actions:
                1) Find all possible moves for piece type
                2) Check if path is obstructed (by same coloured pieces)
                3) *KING Only* check if movement puts King in Check
        '''
        possible_actions = []
        
        for row in range(self.size):
            for col in range(self.size):
                piece = self.board[row][col]
                
                if piece == PAWN:
                    d = 1 if piece.colour == BLACK else -1
                    # Regular moves
                    if self.is_valid_move((row, col), (row+d, col + d)):
                        possible_actions.append((row, col), (row+d, col+d))
                    # Capture moves (Diagonal)
                    if self.is_valid_move(((row, col), (row+d, col+1))):
                        possible_actions.append(((row, col), (row+d, col+1)))
                    if self.is_valid_move(((row, col), (row+d, col-1))):
                        possible_actions.append(((row, col), (row+d, col-1)))

                elif piece == KNIGHT:
                    # L-Movements
                    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                    for dr, dc in knight_moves:
                        new_row, new_col = row + dr, col + dc
                        if self.is_valid_move((row, col), (new_row, new_col)):
                            possible_actions.append(((row, col), (new_row, new_col)))

                elif piece == BISHOP:
                    # Diagonal Movements
                    for i in range(1, self.size):
                        for dr, dc in [(-i, -i), (-i, i), (i, -i), (i, i)]:
                            new_row, new_col = row + dr, col + dc
                            if self.is_valid_move((row, col), (new_row, new_col)): 
                                possible_actions.append(((row, col), (new_row, new_col)))
                
                elif piece == ROOK:
                    # Straight Movements
                    for r in range(self.size):
                        if (r != row) and self.is_valid_move((row, col), (r, col)):
                            possible_actions.append(((row, col), (r, col)))
                    for c in range(self.size):
                        if (c != col) and self.is_valid_move((row, col), (r, col)):
                            possible_actions.append(((row, col), (row, c)))

                elif piece == QUEEN:
                    # Straight Movements
                    for r in range(self.size):
                        if (r != row) and self.is_valid_move((row, col), (r, col)):
                            possible_actions.append(((row, col), (r, col)))
                    for c in range(self.size):
                        if (c != col) and self.is_valid_move((row, col), (r, col)):
                            possible_actions.append(((row, col), (row, c)))
                    # Diagonal Movements
                    for i in range(1, self.size):
                        for dr, dc in [(-i, -i), (-i, i), (i, -i), (i, i)]:
                            new_row, new_col = row + dr, col + dc
                            if self.is_valid_move((row, col), (new_row, new_col)):
                                possible_actions.append(((row, col), (new_row, new_col)))

                elif piece == KING:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if ((dr, dc) == (0, 0)):
                                continue
                            new_row, new_col = row+dr, col+dc
                            if (self.is_valid_move((row, col), (new_row, new_col))): 
                                possible_actions.append(((row, col), (new_row, new_col)))
        
        return possible_actions

    def calculate_reward(state):
        # TODO: implement rewards system
        return 10

    def move_piece(self, action):
        start = action[0]
        end = action[1]
        if self.is_valid_move(start, end):
            piece = self.board[start[0]][start[1]]
            self.board[end[0]][end[1]] = piece
            self.board[start[0]][start[1]] = '.'
            return True
        return False
    
    def is_valid_move(self, start, end):

        start_p = self.board[start[0]][start[1]]
        end_p = self.board[end[0]][end[1]]

        # Check if the end position is within bounds (on the board)
        if not (0 <= end[0] < self.size and 0 <= end[1] < self.size):
            return False
        
        # Check if the end position is occupied by the same color
        if end_p != 0 and ((start_p>0 and end_p>0) or (start_p<0 and end_p<0)):
            return False
        
        p_type = abs(start_p)

        # Check for obstructions, if piece can move multiple squares
        if p_type == ROOK:
            if start[0] == end[0]:  # Horizontal move
                return all(self.board[start[0]][i] == 0 for i in range(min(start[1], end[1]) + 1, max(start[1], end[1])))
            if start[1] == end[1]:  # Vertical move
                return all(self.board[i][start[1]] == 0 for i in range(min(start[0], end[0]) + 1, max(start[0], end[0])))

        elif p_type == BISHOP:
            if abs(start[0] - end[0]) == abs(start[1] - end[1]):  # Diagonal move
                direction_x = 1 if end[0] > start[0] else -1
                direction_y = 1 if end[1] > start[1] else -1
                return all(self.board[start[0] + i * direction_x][start[1] + i * direction_y] == 0 for i in range(1, abs(start[0] - end[0])))
                    
        elif p_type == QUEEN:
            # Check if the move is a valid rook move
            if start[0] == end[0]:  # Horizontal move
                return all(self.board[start[0]][i] == 0 for i in range(min(start[1], end[1]) + 1, max(start[1], end[1]))) and \
                    (end_piece == 0 or end_piece.colour != start_piece.colour)
            if start[1] == end[1]:  # Vertical move
                return all(self.board[i][start[1]] == 0 for i in range(min(start[0], end[0]) + 1, max(start[0], end[0]))) and \
                    (end_piece == 0 or end_piece.colour != start_piece.colour)
            # Check if the move is a valid bishop move
            if abs(start[0] - end[0]) == abs(start[1] - end[1]):
                return all(self.board[start[0] + i * (1 if end[0] > start[0] else -1)][start[1] + i * (1 if end[1] > start[1] else -1)] == 0
                        for i in range(1, abs(start[0] - end[0]))) and \
                    (end_piece == 0 or end_piece.colour != start_piece.colour)
    
        return False
    
    def check_winner(self):
        white_king_alive = any(KING_ID in row for row in self.board)
        black_king_alive = any(-KING_ID in row for row in self.board)
        
        if not white_king_alive:
            return True, BLACK
        elif not black_king_alive:
            return True, WHITE
        else:
            return False, None

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_states(self):
        return self._n_states

    @property
    def goal(self):
        return self._goal

    @property
    def to_cell(self):
        return self._to_cell