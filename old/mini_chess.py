'''
    COMP 4010 [Fall 2024]
    Carleton University

    Project Details:
        ~ RL Chess Agent ~
        Date: October 15, 2024

        Group Members:
            Kyle Eng 101192595
            Joshua Olojede 101182941
'''

# Imports
import random
import numpy as np

from enum import Enum

# Constants
SIZE = 6 # = ROWS = COLUMNS

# Enums for Readability
class PieceType(Enum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

class Colour(Enum):
    WHITE = 1
    BLACK = -1

class ChessPiece(Enum):
    def __init__(self, p_type: PieceType, color: Colour):
        self.p_type = p_type
        self.color = color

class CustomChess6x6:
    def __init__(self):
        self.size = 6
        self.reset_board()  # Initialize board

        '''
        
        
        '''
        self.n_states = 0

        
        self.n_actions = 0

    def reset_board (self):
        # The location and pieces on the board represent the state
        self.board = np.zeros((6, 6), dtype=int)

        # Los Alamos Chess Layout
        # Place Black back-rank pieces
        self.board[0] = [ChessPiece(PieceType.ROOK, Colour.BLACK),
                        ChessPiece(PieceType.KNIGHT, Colour.BLACK),
                        ChessPiece(PieceType.QUEEN, Colour.BLACK),
                        ChessPiece(PieceType.KING, Colour.BLACK),
                        ChessPiece(PieceType.KNIGHT, Colour.BLACK),
                        ChessPiece(PieceType.ROOK, Colour.BLACK)]
        
        # Place Black pawns
        self.board[1] = [ChessPiece(PieceType.PAWN, Colour.BLACK) for _ in range(6)]
        
        # Place White pawns
        self.board[4] = [ChessPiece(PieceType.PAWN, Colour.WHITE) for _ in range(6)]
        
        # Place White back-rank pieces
        self.board[5] = [ChessPiece(PieceType.ROOK, Colour.WHITE),
                        ChessPiece(PieceType.KNIGHT, Colour.WHITE),
                        ChessPiece(PieceType.QUEEN, Colour.WHITE),
                        ChessPiece(PieceType.KING, Colour.WHITE),
                        ChessPiece(PieceType.KNIGHT, Colour.WHITE),
                        ChessPiece(PieceType.ROOK, Colour.WHITE)]
        
        # White always starts in Chess
        self.turn = Colour.WHITE    

        return self.board
        # return self.board.copy()
    
    def is_valid_move(self, start, end):

        start_piece = self.board[start[0]][start[1]]
        end_piece = self.board[end[0]][end[1]]

        # Check if the end position is within bounds
        if not (0 <= end[0] < self.size and 0 <= end[1] < self.size):
            return False
        
        # Check if the end position is occupied by the same color
        # Check if the end position is occupied by the same color
        if (end_piece != 0 and start_piece.colour == end_piece.colour):
            return False
        
         # Implement movement rules for each piece
        if start_piece.piece_type == PieceType.PAWN:
            direction = 1 if start_piece.colour == Colour.BLACK else -1
            # Regular move
            if end == (start[0] + direction, start[1]) and end_piece == 0:
                return True
            # Initial double move
            if (start[0] == (1 if start_piece.colour == Colour.BLACK else 4) and 
                    end == (start[0] + direction * 2, start[1]) and end_piece == 0):
                return True
            # Capture moves
            if end == (start[0] + direction, start[1] + 1) and end_piece != 0 and end_piece.colour != start_piece.colour:
                return True
            if end == (start[0] + direction, start[1] - 1) and end_piece != 0 and end_piece.colour != start_piece.colour:
                return True

        elif start_piece.piece_type == PieceType.ROOK:
            if start[0] == end[0]:  # Horizontal move
                return all(self.board[start[0]][i] == 0 for i in range(min(start[1], end[1]) + 1, max(start[1], end[1])))
            if start[1] == end[1]:  # Vertical move
                return all(self.board[i][start[1]] == 0 for i in range(min(start[0], end[0]) + 1, max(start[0], end[0])))

        elif start_piece.piece_type == PieceType.KNIGHT:
            return (abs(start[0] - end[0]), abs(start[1] - end[1])) in [(2, 1), (1, 2)]

        elif start_piece.piece_type == PieceType.BISHOP:
            if abs(start[0] - end[0]) == abs(start[1] - end[1]):
                return all(self.board[start[0] + i * (1 if end[0] > start[0] else -1)][start[1] + i * (1 if end[1] > start[1] else -1)] == 0
                        for i in range(1, abs(start[0] - end[0])))

        elif start_piece.piece_type == PieceType.QUEEN:
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

        elif start_piece.piece_type == PieceType.KING:
            return abs(start[0] - end[0]) <= 1 and abs(start[1] - end[1]) <= 1

        return False

    def move_piece(self, start, end):
        if self.is_valid_move(start, end):
            piece = self.board[start[0]][start[1]]
            self.board[end[0]][end[1]] = piece
            self.board[start[0]][start[1]] = '.'
            return True
        return False

    def check_winner(self):
        white_king = any('K' in row for row in self.board)
        black_king = any('k' in row for row in self.board)
        
        if not white_king:
            return True, Colour.BLACK
        elif not black_king:
            return True, Colour.WHITE
        else:
            return False, None
    
    def get_moves(self, position):
        row, col = position
        piece = self.board[row][col]
        moves = []

        if piece.p_type == PieceType.PAWN:
            direction = 1 if piece.colour == Colour.BLACK else -1
            # Regular move
            if self.is_valid_move((row, col), (row + direction, col)):
                moves.append(position, (row + direction, col))
            # Initial double move
            if (piece.colour == Colour.BLACK and row == 1) or (piece.colour == Colour.WHITE and row == 4):
                if self.is_valid_move((row, col), (row + direction * 2, col)):
                    moves.append(position, (row + direction * 2, col))
            # Capture moves
            for d in [-1, 1]:
                if self.is_valid_move((row, col), (row + direction, col + d)):
                    moves.append(position, (row + direction, col + d))

        elif piece.p_type == PieceType.KNIGHT:
            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.size and 0 <= new_col < self.size and self.is_valid_move((row, col), (new_row, new_col)):
                    moves.append(position, (new_row, new_col))

        elif piece.p_type == PieceType.BISHOP:
            for i in range(1, self.size):
                # Check all four diagonal directions
                for dr, dc in [(-i, -i), (-i, i), (i, -i), (i, i)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < self.size and 0 <= new_col < self.size:
                        if self.is_valid_move((row, col), (new_row, new_col)):
                            moves.append(position, (new_row, new_col))
                        else:
                            break  # Stop if a piece blocks the way
        
        elif piece.p_type == PieceType.ROOK:
            # Add rook movement logic
            for r in range(self.size):
                if r != row and self.is_valid_move((row, col), (r, col)):
                    moves.append(position, (r, col))
            for c in range(self.size):
                if c != col and self.is_valid_move((row, col), (row, c)):
                    moves.append(position, (row, c))

        elif piece.p_type == PieceType.QUEEN:
            # Add queen movement logic (rook and bishop combined)
            # Rook-like moves
            for r in range(self.size):
                if r != row and self.is_valid_move((row, col), (r, col)):
                    moves.append(position, (r, col))
            for c in range(self.size):
                if c != col and self.is_valid_move((row, col), (row, c)):
                    moves.append(position, (row, c))
            # Bishop-like moves
            for i in range(1, self.size):
                for dr, dc in [(-i, -i), (-i, i), (i, -i), (i, i)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < self.size and 0 <= new_col < self.size:
                        if self.is_valid_move((row, col), (new_row, new_col)):
                            moves.append(position, (new_row, new_col))
                        else:
                            break  # Stop if a piece blocks the way

        elif piece.p_type == Piece.KING:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if (dr, dc) != (0, 0):  # Skip the (0, 0) move
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.size and 0 <= new_col < self.size and self.is_valid_move((row, col), (new_row, new_col)):
                            moves.append(position, (new_row, new_col))
        
        return moves

    def get_possible_actions(self):
        possible_actions = []

        for row in range(self.size):
            for col in range(self.size):
                piece = self.board[row][col]
                if (isinstance(piece, ChessPiece) and (piece.colour == self.turn)):
                    moves = self.get_moves((row, col))
                    possible_actions.append(moves)
        
        return possible_actions
  
    def display(self):
        for row in self.board:
            print(' '.join(row))
        print()

def play_chess(board):
    valid_moves = []
    for i in range(board.size):
        for j in range(board.size):
            if board.board[i][j] != '.':
                piece = board.board[i][j]
                # Check if it's the current player's turn
                if (board.turn == 'white' and piece.islower()) or (board.turn == 'black' and piece.isupper()):
                    continue  # Skip if it's the opponent's piece

                # Check all possible moves (basic example)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if (di != 0 or dj != 0):  # Not the same position
                            new_i = i + di
                            new_j = j + dj
                            # Ensure new_i and new_j are within bounds
                            if 0 <= new_i < board.size and 0 <= new_j < board.size:
                                if board.is_valid_move((i, j), (new_i, new_j)):
                                    valid_moves.append(((i, j), (new_i, new_j)))

    if valid_moves:
        move = random.choice(valid_moves)
        board.move_piece(move[0], move[1])
        print(f'{board.turn.capitalize()} moved: {move[0]} to {move[1]}')
        # Switch turns after the move
        board.turn = 'black' if board.turn == 'white' else 'white'
    else:
        print('No valid moves available for', board.turn)
