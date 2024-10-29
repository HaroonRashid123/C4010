import random

class CustomChess6x6:
    def __init__(self):
        self.size = 6
        self.board = [["." for _ in range(self.size)] for _ in range(self.size)]
        self.setup_pieces()
        self.turn = "white"  # Start with White's turn

    def setup_pieces(self):
        '''
            R = Rook
            B = Bishop
            H = Knight(Horse)
            Q = Queen
            K = King
            P = Pawn
        '''
        # Initialize pieces on the board
        self.board[0] = ["R", "N", "B", "Q", "K", "R"]  # Black pieces
        self.board[1] = ["P"] * 6                      # Black pawns
        self.board[4] = ["p"] * 6                      # White pawns
        self.board[5] = ["r", "n", "b", "q", "k", "r"]  # White pieces

    def display(self):
        for row in self.board:
            print(" ".join(row))
        print()

    def is_valid_move(self, start, end):
        start_piece = self.board[start[0]][start[1]]
        end_piece = self.board[end[0]][end[1]]

        # Check if the end position is within bounds
        if not (0 <= end[0] < self.size and 0 <= end[1] < self.size):
            return False
        
        # Check if the end position is occupied by the same color
        if (end_piece.islower() and start_piece.islower()) or \
           (end_piece.isupper() and start_piece.isupper()):
            return False
        
        # Implement movement rules for each piece
        if start_piece.upper() == "P":  # Pawn
            if start_piece == "P":  # Black pawn
                if end == (start[0] + 1, start[1]) and end_piece == ".":
                    return True
                if (start[0] == 1 and end == (start[0] + 2, start[1]) and end_piece == "."):
                    return True
                if end == (start[0] + 1, start[1] + 1) and end_piece.islower():
                    return True
                if end == (start[0] + 1, start[1] - 1) and end_piece.islower():
                    return True
            else:  # White pawn
                if end == (start[0] - 1, start[1]) and end_piece == ".":
                    return True
                if (start[0] == 4 and end == (start[0] - 2, start[1]) and end_piece == "."):
                    return True
                if end == (start[0] - 1, start[1] + 1) and end_piece.isupper():
                    return True
                if end == (start[0] - 1, start[1] - 1) and end_piece.isupper():
                    return True

        if start_piece.upper() == "R":  # Rook
            if start[0] == end[0]:  # Horizontal move
                return all(self.board[start[0]][i] == "." for i in range(min(start[1], end[1]) + 1, max(start[1], end[1])))
            if start[1] == end[1]:  # Vertical move
                return all(self.board[i][start[1]] == "." for i in range(min(start[0], end[0]) + 1, max(start[0], end[0])))

        if start_piece.upper() == "N":  # Knight
            return (abs(start[0] - end[0]), abs(start[1] - end[1])) in [(2, 1), (1, 2)]

        if start_piece.upper() == "B":  # Bishop
            if abs(start[0] - end[0]) == abs(start[1] - end[1]):
                return all(self.board[start[0] + i * (1 if end[0] > start[0] else -1)][start[1] + i * (1 if end[1] > start[1] else -1)] == "."
                           for i in range(1, abs(start[0] - end[0])))

        if start_piece.upper() == "Q":  # Queen
            # Check if the move is a valid rook move
            if start[0] == end[0]:  # Horizontal move
                return all(self.board[start[0]][i] == "." for i in range(min(start[1], end[1]) + 1, max(start[1], end[1]))) and \
                       (self.board[start[0]][end[1]] == "." or self.board[start[0]][end[1]].isupper() != start_piece.isupper())
            if start[1] == end[1]:  # Vertical move
                return all(self.board[i][start[1]] == "." for i in range(min(start[0], end[0]) + 1, max(start[0], end[0]))) and \
                       (self.board[end[0]][start[1]] == "." or self.board[end[0]][start[1]].isupper() != start_piece.isupper())
            # Check if the move is a valid bishop move
            if abs(start[0] - end[0]) == abs(start[1] - end[1]):
                return all(self.board[start[0] + i * (1 if end[0] > start[0] else -1)][start[1] + i * (1 if end[1] > start[1] else -1)] == "."
                           for i in range(1, abs(start[0] - end[0]))) and \
                       (self.board[end[0]][end[1]] == "." or self.board[end[0]][end[1]].isupper() != start_piece.isupper())

        if start_piece.upper() == "K":  # King
            return abs(start[0] - end[0]) <= 1 and abs(start[1] - end[1]) <= 1

        return False

    def move_piece(self, start, end):
        if self.is_valid_move(start, end):
            piece = self.board[start[0]][start[1]]
            self.board[end[0]][end[1]] = piece
            self.board[start[0]][start[1]] = "."
            return True
        return False

    def check_winner(self):
        white_king = any("K" in row for row in self.board)
        black_king = any("k" in row for row in self.board)

        if not white_king:
            return "Black wins!"
        if not black_king:
            return "White wins!"
        return None

def play_chess(board):
    valid_moves = []
    for i in range(board.size):
        for j in range(board.size):
            if board.board[i][j] != ".":
                piece = board.board[i][j]
                # Check if it's the current player's turn
                if (board.turn == "white" and piece.islower()) or (board.turn == "black" and piece.isupper()):
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
        print(f"{board.turn.capitalize()} moved: {move[0]} to {move[1]}")
        # Switch turns after the move
        board.turn = "black" if board.turn == "white" else "white"
    else:
        print("No valid moves available for", board.turn)

# Main loop to demonstrate integration
custom_board = CustomChess6x6()
custom_board.display()

while True:  # Run until there's a winner
    play_chess(custom_board)
    custom_board.display()
    
    winner = custom_board.check_winner()
    if winner:
        print(winner)
        break
