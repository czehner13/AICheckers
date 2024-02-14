import argparse
import configparser
from copy import deepcopy
from time import time
import math

# Globals
maxUtility = 1e9
blackTurn = True
maxTime = 10
#5 takes an average of 
maxDepth = 3
#STEP_THRESHOLD = 3;
playerType = None
boardSize = None
positions = None


class checkerState:
    def __init__(self, boardState, blackNextTurn, moves):
        self.boardState = boardState
        self.blackNextTurn = blackNextTurn
        self.moves = moves
        #self.whiteOnBoard =   # Hops taken by a disc to reach the current state

    # This methods walks cell by cell to check what pieces are on the board.
    # The method returns 'True' only when
    #   There is atleast one white disk and no black disk
    #   There is atleast one black disk and no white disk
    # else it returns false.
    def terminalState(self):
        blackOnBoard, whiteOnBoard = False, False
        for row in self.boardState:
            for cell in row:
                # if only black pieces are found
                if (cell == 'b' or cell == 'B'):
                    blackOnBoard = True
                # if only white pieces are found
                elif (cell == 'w' or cell == 'W'):
                    whiteOnBoard = True
                # if white and black pieces are found
                if (blackOnBoard and whiteOnBoard):
                    return False
        self.hasBlack = blackOnBoard
        return True
        

    # A simple terminal function that returns positive value when there are
    # opponent pieces on board.
    # returns negative value when there are no opponent pieces
    def terminalUtility(self):
        if (player == 'b' and self.hasBlack) or (player == 'w' and not self.blackOnBoard):
            return maxUtility
        else:
            return -maxUtility
        # return maxUtility if blackTurn != self.isLoserBlack else -maxUtility

    # Setting Boundaries and Moves/Jumps for each piece
    def getSuccessors(self):
        def generateSteps(cell):
            steps = []
            # White moves up, Black moves down
            whiteAvailableMoves, blackAvailableMoves = [(-1, -1), (-1, 1)], [(1, -1), (1, 1)]
            kingMoves = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            #if cell != 'b': steps.extend(whiteAvailableMoves)
            #if cell != 'w': steps.extend(blackAvailableMoves)

            if cell == 'w': steps.extend(whiteAvailableMoves)
            if cell == 'b': steps.extend(blackAvailableMoves)

            if cell == 'W': steps.extend(kingMoves)
            if cell == 'B': steps.extend(kingMoves)

            return steps

        def generateMoves(board, i, j, successors):
            for step in generateSteps(board[i][j]):
                # Set coordinates for move
                x, y = i + step[0], j + step[1]
                # Check if still in bounds
                if (x >= 0 and x < boardSize) and (y >= 0 and y < boardSize) and board[x][y] == '_':
                    # Copy board and set new coordinates/empty space where piece was
                    boardCopy = deepcopy(board)
                    # Replace moved piece position with '_'
                    boardCopy[x][y], boardCopy[i][j] = boardCopy[i][j], '_'
                    # A pawn is promoted when it reaches the last row
                    if (x == (boardSize - 1) and self.blackNextTurn) or (x == 0 and not self.blackNextTurn):
                        boardCopy[x][y] = boardCopy[x][y].upper()
                    successors.append(checkerState(boardCopy, not self.blackNextTurn, [(i, j), (x, y)]))

        def generateJumps(board, i, j, moves, successors):
            isJumpOver = True
            # Deepcopy board and moves to simulate moves made before committing
            boardCopy, movesCopy = deepcopy(board), deepcopy(moves)
            for step in generateSteps(board[i][j]):
                # Begin copy steps
                x, y = i + step[0], j + step[1]
                if (x >= 0 and x < boardSize) and (y >= 0 and y < boardSize) and board[x][y] != '_' and board[i][j].lower() != board[x][
                    y].lower():
                    # Save cell space for after jump
                    xp, yp = x + step[0], y + step[1]
                    # Check if still in bounds
                    if xp >= 0 and xp < boardSize and yp >= 0 and yp < boardSize and board[xp][yp] == '_':
                        board[xp][yp], save = board[i][j], board[x][y]
                        # Replace moved piece position with '_'
                        board[i][j] = board[x][y] = '_'
                        previous = board[xp][yp]
                        # A pawn is promoted when it reaches the last row
                        if (xp == boardSize - 1 and self.blackNextTurn) or (xp == 0 and not self.blackNextTurn):
                            board[xp][yp] = board[xp][yp].upper()

                        # Applying copied moves to actual board
                        moves.append((xp, yp))
                        generateJumps(board, xp, yp, moves, successors)
                        # Removes the single-move, store only the jump position
                        moves.pop()
                        # Store all states
                        board[i][j], board[x][y], board[xp][yp] = previous, save, '_'
                        # Iterate again
                        isJumpOver = False
            if isJumpOver and len(moves) > 1:
                # Update successors for new postion
                successors.append(checkerState(boardCopy, not self.blackNextTurn, movesCopy))

        playerTurn = 'b' if self.blackNextTurn else 'w'
        successors = []
        
        #Generate move strategy: if there are jumps choose jumps first
        # generate jumps
        for i in range(boardSize):
            for j in range(boardSize):
                if self.boardState[i][j].lower() == playerTurn:
                    generateJumps(self.boardState, i, j, [(i, j)], successors)
        if len(successors) > 0: return successors

        # generate moves
        for i in range(boardSize):
            for j in range(boardSize):
                if self.boardState[i][j].lower() == playerTurn:
                    generateMoves(self.boardState, i, j, successors)
        return successors


# Setup configs
# get the input from a properties file
def setup():
    global playerType
    global boardSize
    global positions
    global turns

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Location of the config file")

    args = parser.parse_args()
    if args.config:
        print("Using config file {} to load properties.".format(args.config))
    else:
        # default location
        args.config = "config.ini"
        print("Using config file {} to load properties".format(args.config))

    config = configparser.ConfigParser()
    config.read(args.config)

    playerType = config['default']['player.type']
    boardSize = int(config['default']['board.size'])
    #boardSize = 8
    turns = int(config['default']['player.turns'])

    # initialize the list
    # positions = {new_list: [] for new_list in range(boardSize)}
    positions = []
    for i in range(boardSize):
        # positions = list(config['positions'][str(i+1)])
        positions.append(list(config['positions'][str(i + 1)]))
        
    #print("Input Grid is")
    printGrid(positions)


def heuristic(state):

    h1 = pieces_calculation_heuristic(state)
    h2 = round(escape_heuristic(state), 2)
    h3 = round(distToKing(state), 2)
    #print(" Heuristic value for all the pieces on the board is: {}".format(h1))
    #Early game: "aggressive"; Late game: "Conservative"
    
    if gameStrategy(state) == "conservative":
        alpha,beta,gamma = 1.5,0.01,0.01
        heuristic = alpha*h1 + beta*h2 + gamma*h3

    if gameStrategy(state) == "aggresive":
        alpha,beta,gamma = 1.5,0.005,0.015
        heuristic = alpha*h1 + beta*h2 + gamma*h3
        
    return heuristic

def printHeuristic(state):

    h1 = pieces_calculation_heuristic(state)
    h2 = round(escape_heuristic(state), 2)
    h3 = round(distToKing(state), 2)
    #print(" Heuristic value for all the pieces on the board is: {}".format(h1))
    #Early game: "aggressive"; Late game: "Conservative"
    
    if gameStrategy(state) == "conservative":
        alpha,beta,gamma = 1,0.015,0.005
        heuristic = alpha*h1 + beta*h2 + gamma*h3
        return [alpha*h1 , beta*h2 , gamma*h3]

    if gameStrategy(state) == "aggresive":
        alpha,beta,gamma = 1,0.01,0.01
        heuristic = alpha*h1 + beta*h2 + gamma*h3
        return [alpha*h1 , beta*h2 , gamma*h3]
        
    #return heuristic

#when our piece are much less than the oppoenent, run as far as possible
def escape_heuristic(state):
    #calculate distance:
    #for black
    if player[0] == "b":
        dist = 0
        for r in range(8):
            for c in range(8):
                if state.boardState[r][c] == "w" or state.boardState == "W":
                    dist += cal_dist(r, c, state)
    #for white               
    if player[0] == "w":
        dist = 0
        for r in range(8):
            for c in range(8):
                if state.boardState[r][c] == "b" or state.boardState == "B":
                    dist += cal_dist(r, c, state)
    return dist


#cal dist between single piece(x, y) to all other opponent pieces
def cal_dist(x, y, state):
    #for balck:
    if player[0] == "b":
        dist = 0
        for r in range(8):
            for c in range(8):
                if state.boardState[r][c] == "b" or "B" :
                    dist += math.sqrt((r -x) * (r - x) + (c - y) * (c - y))
    #for white:
    if player[0] == "w":
        dist = 0
        for r in range(8):
            for c in range(8):
                if state.boardState[r][c] == "w" or "W" :
                    dist += math.sqrt((r -x) * (r - x) + (c - y) * (c - y))
    return dist

#cal dist between single piece(x, y) to King pieces only
def cal_distKing(x, y, state):
    #for balck:
    if player[0] == "b":
        dist = 0
        for r in range(8):
            for c in range(8):
                if state.boardState[r][c] == "B" :
                    dist += math.sqrt((r -x) * (r - x) + (c - y) * (c - y))
        return dist
    #for white:
    if player[0] == "w":
        dist = 0
        for r in range(8):
            for c in range(8):
                if state.boardState[r][c] == "W" :
                    dist += math.sqrt((r -x) * (r - x) + (c - y) * (c - y))
        return dist

def prioritizeKing(state):
    blackKingCount, whiteKingCount, kingCount = 0, 0, 0
    for row in state.boardState:
        for cell in row:
            if player[0] == "w":
                if cell == "B":
                    blackKingCount += 1
            if player[0] == "b":
                if cell == "W":
                    whiteKingCount += 1
    if player[0] == 'b':
        kingCount = whiteKingCount
    else:
        kingCount = blackKingCount
    return kingCount
    
def distToKing(state):
    kingCount = prioritizeKing(state)
    if kingCount > 0:
        if player[0] == 'b':
            dist = 0
            for r in range(8):
                for c in range(8):
                    dist += cal_distKing(r, c, state)
            return dist
        if player[0] == "w":
            dist = 0
            for r in range(8):
                for c in range(8):
                    dist += cal_distKing(r, c, state)
            return dist
    else: 
        return 0

# Returns sum of ours and theirs pieces
def pieces_calculation_heuristic(state):
    # 1 for a normal piece, 1.5 for a king
    black, white = 0, 0
    r, c = 0, 0
    for row in state.boardState:
        for cell in row:
            # increment column count
            c += 1
            if cell == 'b':
                black += 1.0 * (r + 1)
                black += attack_heuristic(cell, r, c, state)
            elif cell == 'B':
                black += 1.5 * (r + 1)
                black += attack_heuristic(cell, r, c, state)
            elif cell == 'w':
                white += 1.0 * (boardSize - r + 1)
                white += attack_heuristic(cell, r, c, state)
            elif cell == 'W':
                white += 1.5 * (boardSize - r + 1)
                white += attack_heuristic(cell, r, c, state)
        # increment row count
        r += 1
        c = 0
    return black - white if blackTurn else white - black

# function that calculates whether there are opponent's pieces that can attack
def attack_heuristic(piece, row, column, state):

    result = 0
    # if black check for attacks from white
    if player[0] == 'b':
        #check attack from right
        ri = row + 1
        rj = column + 1
        # check bounds
        if (ri >= 0) and (ri <= boardSize - 1) and (rj >= 0) and (rj <= boardSize - 1):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'w' or state.boardState[ri][rj] == 'W':
                result = -5 * check_consecutive_pieces(piece, ri, rj, state, "right")

        # check attack from left
        ri = row + 1
        rj = column - 1
        # check bounds
        if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'w' or state.boardState[ri][rj] == 'W':
                result = -5 * check_consecutive_pieces(piece, ri, rj, state, "left")

        # check attack from back right (white king)
        ri = row + 1
        rj = column + 1
        # check bounds
        if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'W':
                result = -5

        # check attack from back left (white king)
        ri = row - 1
        rj = column + 1
        # check bounds
        if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'W':
                result = -5

    if player[0] == 'w':
        # if white check for attacks from black
        # check attack from right
        ri = row - 1
        rj = column + 1
        # check bounds
        if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'b' or state.boardState[ri][rj] == 'B':
                result = -5

        # check attack from left
        ri = row - 1
        rj = column - 1
        # check bounds
        if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'b' or state.boardState[ri][rj] == 'B':
                result = -5

                # check attack from back right king
        ri = row + 1
        rj = column + 1
        # check bounds
        if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'B':
                result = -5

        # check attack from left king
        ri = row + 1
        rj = column - 1
        # check bounds
        if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
            # are we getting eaten ?
            if state.boardState[ri][rj] == 'B':
                result = -5

    return result

# check if our pieces can be eaten in an sequence
def check_consecutive_pieces(piece, row, column, state, direction):
    result = 1
    if piece.lower() == 'b':
        # check left sequence
        if direction.lower() == "left":
            ri = row - 1
            rj = column - 1
            # check bounds
            if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
                # are we getting eaten ?
                if state.boardState[ri][rj].lower() == 'b':
                    # recursively check for consicutive pieces
                    result = 5 * check_consecutive_pieces(piece, ri, rj, state, direction)
        if direction.lower() == "right":
            ri = row - 1
            rj = column + 1
            # check bounds
            if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
                # are we getting eaten ?
                if state.boardState[ri][rj].lower() == 'b':
                    # recursively check for consicutive pieces
                    result = 5 * check_consecutive_pieces(piece, ri, rj, state, direction)
    if piece.lower() == 'w':
        # check left sequence
        if direction.lower() == "left":
            ri = row + 1
            rj = column - 1
            # check bounds
            if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
                # are we getting eaten ?
                if state.boardState[ri][rj].lower() == 'w':
                    # recursively check for consicutive pieces
                    result = 5 * check_consecutive_pieces(piece, ri, rj, state, direction)
        if direction.lower() == "right":
            ri = row + 1
            rj = column + 1
            # check bounds
            if (ri >= 0) and (ri <= 7) and (rj >= 0) and (rj <= 7):
                # are we getting eaten ?
                if state.boardState[ri][rj].lower() == 'w':
                    # recursively check for consicutive pieces
                    result = 5 * check_consecutive_pieces(piece, ri, rj, state, direction)

    return result

#Game strategy: "conservative" or "aggresive"
def gameStrategy(state):
    #count num of black and white pieces
    strategy = "aggresive"
    black_count, white_count = 0, 0
    #when opponent has 2 more than our pieces, become "conservative"
    for row in state.boardState:
        for cell in row:
            if cell in {"b", "B"}:
                black_count += 1
            if cell in {"w", "W"}:
                white_count += 1

    #When it is time to be conservative:
    if player[0] == "w":
        if (black_count - white_count) < -2:
            strategy = "conservative"


    if player[0] == "b":
        if (white_count - black_count) < -2:
            strategy = "conservative"

    return strategy


def minMax(state, heuristic):

    startTime = time()

    def alphaBetaSearch(state, alpha, beta, depth):
        def alphaPrune(state, alpha, beta, depth):
            # Negative 'infinity' to test against
            minVal = -maxUtility
            for successor in state.getSuccessors():
                # Set minVal to highest child
                minVal = max(minVal, alphaBetaSearch(successor, alpha, beta, depth))
                # No prune
                #if minVal >= beta: return minVal
                alpha = max(alpha, minVal)
            return minVal

        def betaPrune(state, alpha, beta, depth):
            # Positive 'infinity' to test against
            maxVal = maxUtility
            for successor in state.getSuccessors():
                # Set maxVal to lowest child
                maxVal = min(maxVal, alphaBetaSearch(successor, alpha, beta, depth - 1))
                # No prune
                #if maxVal <= alpha: return maxVal
                beta = min(beta, maxVal)
            return maxVal

        if state.terminalState(): return state.terminalUtility()
        if depth <= 0 or time() - startTime > maxTime: return heuristic(state)
        return alphaPrune(state, alpha, beta, depth) if state.blackNextTurn == blackTurn else betaPrune(state, alpha,
                                                                                                       beta, depth)

    bestMove = None
    # Search tree within n-depth
    for depth in range(1, maxDepth):
        if time() - startTime > maxTime: break
        val = -maxUtility
        for successor in state.getSuccessors():
            # Update score
            score = alphaBetaSearch(successor, -maxUtility, maxUtility, depth)
            if score > val:
                val, bestMove = score, successor.moves
    if(time()-startTime >= maxTime):
        print("Time has exceeded the maximum time: {} seconds, outputting best move found before time ran out.".format(maxTime))
    else:
        print("Time taken: {} seconds".format(time()-startTime))
    return bestMove


def alphaBetaMinMax(state, heuristic):
    startTime = time()

    def alphaBetaSearch(state, alpha, beta, depth):
        def alphaPrune(state, alpha, beta, depth):
            # Negative 'infinity' to test against
            minVal = -maxUtility
            for successor in state.getSuccessors():
                # Set minVal to highest child
                minVal = max(minVal, alphaBetaSearch(successor, alpha, beta, depth))
                # Prune
                if minVal >= beta: return minVal
                alpha = max(alpha, minVal)
            return minVal

        def betaPrune(state, alpha, beta, depth):
            # Positive 'infinity' to test against
            maxVal = maxUtility
            for successor in state.getSuccessors():
                # Set maxVal to lowest child
                maxVal = min(maxVal, alphaBetaSearch(successor, alpha, beta, depth - 1))
                # Prune
                if maxVal <= alpha: return maxVal
                beta = min(beta, maxVal)
            return maxVal

        if state.terminalState(): return state.terminalUtility()
        if depth <= 0 or time() - startTime > maxTime: return heuristic(state)
        return alphaPrune(state, alpha, beta, depth) if state.blackNextTurn == blackTurn else betaPrune(state, alpha,
                                                                                                       beta, depth)

    bestMove = None
    # Search tree within n-depth
    for depth in range(1, maxDepth):
        if time() - startTime > maxTime: break
        val = -maxUtility
        for successor in state.getSuccessors():
            # Update score
            score = alphaBetaSearch(successor, -maxUtility, maxUtility, depth)
            if score > val:
                val, bestMove = score, successor.moves
    if(time()-startTime >= maxTime):
        print("Time has exceeded the maximum time: {} seconds, outputting best move found before time ran out.".format(maxTime))
    else:
        print("Time taken: {} seconds".format(time()-startTime))
  
    print("h1: {}".format(printHeuristic(state)[0]))
    print("h2: {}".format(round(printHeuristic(state)[1])))
    print("h3: {}".format(round(printHeuristic(state)[2])))
    print("Max Heuristic: {}".format(val))
    return bestMove


# Global method to get resulting grid after one step;
def getGrid(grid, move, blackTurn):
    #First determine which palaye's turn: b or w.
    #movePlayer: player to move;
    #delPlayer: When "hop" happens, we need to remove pices hopped/eaten. 
    if blackTurn:
        movPlayer = 'b';
        delPlayer = 'w'
    else:
        movPlayer = 'w'
        delPlayer = 'b';
    #Get initStep location and lastStep location

    # Render King
    for k,v in move:
        if k >= (boardSize - 1) or k <= 0:
            movPlayer = movPlayer.upper()
            print("We have a new king {} at {}".format(movPlayer, (k, v)))


    initStep = move[0];
    lastStep = move[len(move) - 1];
    for i in range(1, len(move), 1):
        prevState = move[i - 1];
        diffX = prevState[0] - move[i][0];
        # If got a 'hop', simply remove the hopped pieces;
        if ((diffX != 1) and (diffX != -1)):
            print("Hey, hop!")
            # cal coords of pieces hopped;
            eatenX = (prevState[0] + move[i][0]) / 2;
            eatenY = (prevState[1] + move[i][1]) / 2;
            x = int(eatenX);
            y = int(eatenY);
            # modifying eaten rows:
            tempList = list(grid[x]);
            tempList[y] = "_";
            newList = tempList;
            grid[x] = newList;
            print("{}th line is: {} ".format(x, grid[x]));

    # modify iniStep;
    #grid is string, need to transfer to list, get modified and transfor back
    print("Output grid is:")
    gridList = list(grid[initStep[0]]);
    movePiece = gridList[initStep[1]]
    gridList[initStep[1]] = "_";
    newList = gridList;
    grid[initStep[0]] = newList;
    # print(grid[initStep[0]]);
    # modify lastStep:
    #if lastStep[0] == 0 or 7:
    gridList = list(grid[lastStep[0]]);
    #"King" rendering:
    #print ("grid: {}".format(grid))
    gridList[lastStep[1]] = movPlayer.upper() if lastStep[0] == (0 or 7) else movePiece;
    newList = gridList;
    grid[lastStep[0]] = newList;
    #add column/row number and row number when displayed:
    print("     0    1    2    3    4    5    6    7  ")
    i = 0
    for row in grid:
        print("{}: {}".format(i, row))
        #print(row);
        i +=1
    return 0;

def printGrid(grid):
    print("Output grid is:")
    # add column/row number and row number when displayed:
    print("     0    1    2    3    4    5    6    7  ")
    i = 0
    for row in grid:
        print("{}: {}".format(i, row))
        # print(row);
        i += 1
    print("\n")


def playGames(position, blackTurn):
    state = checkerState(position, blackTurn, [])
    availableMoves = []
    availablePiece = {}
    #It is human's turn:
    if state.blackNextTurn:
        # Find available moves for player
        availableMoves = state.getSuccessors()
        # Build dictionary for moves
        for i in range(len(availableMoves)):
            availablePiece.update({i:availableMoves[i].moves})
        for j in availablePiece:
            print("{}: {}".format(j, availablePiece[j]))
        # Player chooses dictionary index
        playerMove = int(input("Select a move you would like to make: "))
        print(playerMove)
        if playerMove in availablePiece.keys():
            # Move piece according to input
            move = availablePiece[playerMove]
            print (move)
            # Output
            getGrid(position, move, blackTurn)
            return position
       # Error-checking input
        else:
            print("Please choose a number in range: " + str(range(i)))
            playerMove = int(input("Select a move you would like to make: "))
            # Move piece according to input
            move = availablePiece[playerMove]
            print (move)
            # Output
            getGrid(position, move, blackTurn)
            return position
    #else: It is bot's turn
    else:
        print("Wait a moment, let me think how to beat you")
        # Apply alpha-beta min-max algorithm
        move = alphaBetaMinMax(state, heuristic)
        print("AI is now playing: {}".format(gameStrategy(state)))
        #move = minMax(state, heuristic)
        getGrid(position, move, blackTurn)
        return position

if __name__ == '__main__':
    setup()
    player = playerType
    
    #boardSize = boardSize
    turns = turns


    blackTurn = player[0] == 'b'
    state = checkerState(positions, blackTurn, [])


    t = 0
    while t < turns:
        playGames(positions, blackTurn)
        blackTurn = not blackTurn
        t += 1

    '''
    state = checkerState(positions, blackTurn, [])
    move = alphaBetaMinMax(state, heuristic)
    getGrid(positions, move, blackTurn);
    '''

    # for step in move:
    #    print(step[0], step[1])