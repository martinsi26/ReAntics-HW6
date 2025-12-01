import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *
import pickle


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "HW6")
        # Dictionary to store how many times each category was visited
        # Key = category tuple, Value = visit count (int)
        self.categoryVisits = {}

        # Track utility for each category (key = category tuple, value = float)
        self.utilityTable = {}

        # Stores the previous state for Bellman Equation
        self.prevState = None

        # GLIE parameters
        self.epsilon = 0.5       # initial exploration probability
        self.epsilonDecay = 0.99 # decay factor per move
        self.epsilonMin = 0.05   # minimum exploration

        # Stores eligibility traces for visited categories
        self.eligibility = {}  # key = category tuple, value = trace value
        self.lambda_ = 0.8     # decay factor for eligibility traces

        # Load learning if file exists
        import os
        if os.path.exists("utility_table.pkl"):
            with open("utility_table.pkl", "rb") as f:
                self.utilityTable = pickle.load(f)
            print("Loaded utility table from file")

    def saveLearning(self, filename="utility_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.utilityTable, f)
        print(f"Utility table saved to {filename}")

    
    ##
    #getPlacement 
    #
    # The agent uses a hardcoded arrangement for phase 1 to provide maximum
    # protection to the queen.  Enemy food is placed randomly.
    #
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #
    # This agent simply gathers food as fast as it can with its worker.  It
    # never attacks and never builds more ants.  The queen is never moved.
    #
    ##
    def getMove(self, currentState):
        category = self.stateToCategory(currentState)

        # Increment trace for the current state
        self.eligibility[category] = 1.0

        # Decay all other traces
        for s in list(self.eligibility.keys()):
            if s != category:
                self.eligibility[s] *= self.lambda_
            # Remove tiny traces to save space
            if self.eligibility[s] < 0.01:
                del self.eligibility[s]

        # Increment visits
        self.categoryVisits[category] = self.categoryVisits.get(category, 0) + 1
        if category not in self.utilityTable:
            self.utilityTable[category] = 0.0

        # --- Initialize utility if first visit ---
        if category not in self.utilityTable:
            self.utilityTable[category] = 0.0

        # TD update if we have a previous state
        if self.prevState is not None:
            self.updateUtility(self.prevState, currentState)

        # Store current state as previous for next turn
        self.prevState = currentState

        # --- GLIE: epsilon-greedy action selection ---
        legalMoves = listAllLegalMoves(currentState)
        takeRandom = random.random() < self.epsilon

        if takeRandom:
            chosenMove = random.choice(legalMoves)  # explore
        else:
            # exploit: pick move leading to highest expected utility
            bestMove = None
            bestUtility = -float("inf")
            for move in legalMoves:
                # simulate the move
                moveUtility = self.estimateMoveUtility(currentState, move)
                if moveUtility > bestUtility:
                    bestUtility = moveUtility
                    bestMove = move
            chosenMove = bestMove if bestMove is not None else random.choice(legalMoves)

        # Decay epsilon (GLIE)
        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)

        return chosenMove    

    def estimateMoveUtility(self, currentState, move):
        """
        Estimate the utility of a move by simulating its effect
        using getNextState, without requiring full game simulation.
        """

        # Simulate the result of making the move
        simulatedState = getNextState(currentState, move)

        # Convert the simulated state to a category
        s_category = self.stateToCategory(simulatedState)

        # Look up the utility value for that category in the utility table
        return self.utilityTable.get(s_category, 0.0)

    
    ##
    #getAttack
    #
    # This agent never attacks
    #
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[0]  #don't care
        
    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        self.prevState = None
        self.saveLearning()  # saves utilityTable to file

    def reward(self, state):
        me = state.whoseTurn
        enemy = 1 - state.whoseTurn
        winner = getWinner(state)

        if winner == me:
            return 1
        elif winner == enemy:
            return -1
        return -0.001
    
    #TD Learning Bellman Equation
    def updateUtility(self, prevState, currentState, alpha=0.1, discount=0.9):
        """
        Update utility using TD Learning:
        U(s) = U(s) + alpha * [R(s) + discount * U(s') - U(s)]
        
        prevState = s
        currentState  = s'
        """
        s_prime = self.stateToCategory(currentState)
        U_s_prime = self.utilityTable.get(s_prime, 0)
        
        # TD error for this transition
        U_prev = self.utilityTable.get(self.stateToCategory(prevState), 0)
        delta = self.reward(prevState) + discount * U_s_prime - U_prev

        for s, e_s in self.eligibility.items():
            U_s = self.utilityTable.get(s, 0)
            self.utilityTable[s] = U_s + alpha * delta * e_s


    def stateToCategory(self, state):
        def getBin(value, bin):
            for i, b, in enumerate(bin):
                if value < b:
                    return i
            return len(bin)
    
        # --- BINS ---
        foodBin = [0, 2, 4, 6, 8, 10, 11] # Absolute food bins (for AI and enemy)
        foodDiffBin = [-6, -3, -1, 0, 1, 3, 6] # Relative food difference bins
        combatDiffBin = [-3, -1, 0, 1, 3] # Combat power difference bins
        workerBin = [0, 1, 2] # Worker count bins
        queenHPBin = [0.25, 0.5, 0.75, 1.01] # Queen HP bins (0–1 normalized)
        threatBin = [0, 2] # Threat bins (queen/hill)
        foodDistBin = [0, 1, 2, 4, 6] # Nearest food distance bins
        gameStageBins = [0.15, 0.35, 0.6, 0.85] # Stage of the game based on food count

        
        myInv = getCurrPlayerInventory(state)
        enemyInv = getEnemyInv(self, state)

        # --- Food ---
        myFood = myInv.foodCount
        enemyFood = enemyInv.foodCount
        myFoodCat = getBin(myFood, foodBin)
        enemyFoodCat = getBin(enemyFood, foodBin)
        foodDiff = myFood - enemyFood
        foodDiffCat = getBin(foodDiff, foodDiffBin)

        # --- Combat ---
        def combatPower(inv):
            power = 0
            for a in inv.ants:
                if a.type == SOLDIER:
                    power += 1
                elif a.type == R_SOLDIER:
                    power += 1  
                elif a.type == DRONE:
                    power += 1
            return power

        combatDiff = combatPower(myInv) - combatPower(enemyInv)
        combatCat = getBin(combatDiff, combatDiffBin)
        
        # --- Workers ---
        myWorkers = getAntList(state, state.whoseTurn, (WORKER,))
        numWorkers = len(myWorkers)
        workerCat = getBin(numWorkers, workerBin)

        # --- Queen HP ---
        myQueen = myInv.getQueen()
        myQueenHP = myQueen.health
        myQueenHPCat = getBin(myQueenHP, queenHPBin)

        # --- Threats ---
        enemyAnts = enemyInv.ants
        queenThreat = 0
        hillThreat = 0

        hill = myInv.getAnthill()

        for a in enemyAnts:
            if a.type in (SOLDIER, R_SOLDIER, DRONE):
                if a.coords and myQueen.coords and stepsToReach(a.coords, myQueen.coords) <= 2:
                    queenThreat = 1
                if a.coords and hill.coords and stepsToReach(a.coords, hill.coords) <= 2:
                    hillThreat = 1

        queenThreatCat = getBin(queenThreat, threatBin)
        hillThreatCat = getBin(hillThreat, threatBin)

        # --- Nearest Food Distance ---
        foodLocation = getConstrList(state, None, (FOOD,))
        myWorkers = getAntList(state, state.whoseTurn, (WORKER,))

        hill = myInv.getAnthill()
        tunnel = myInv.getTunnels()

        # Start distances very large
        bestDistToFood = 999
        bestDistToDropoff = 999

        for w in myWorkers:
            if not w.coords:
                continue

            if not w.carrying:
                # Worker going to food
                for food in foodLocation:
                    d = stepsToReach(state, w.coords, food.coords)
                    if d < bestDistToFood:
                        bestDistToFood = d

            else:
                # Worker returning food → choose nearest drop-off
                dHill = stepsToReach(state, w.coords, hill.coords)
                dTun  = stepsToReach(state, w.coords, tunnel[0].coords)
                bestDistToDropoff = min(bestDistToDropoff, dHill, dTun)

        # Choose best overall “food efficiency”
        # Prefer workers carrying food (immediate scoring potential)
        bestWorkerDist = min(bestDistToFood, bestDistToDropoff)

        # Convert to category
        foodDistCat = getBin(bestWorkerDist, foodDistBin)

        # ---------- GAME STAGE ----------
        myProgress = myInv.foodCount / 11.0
        enemyProgress = enemyInv.foodCount / 11.0
        
        gameProgress = max(myProgress, enemyProgress)
        gameStageCat = getBin(gameProgress, gameStageBins)

        # --- Return tuple ---
        category = (
            myFoodCat,
            enemyFoodCat,
            foodDiffCat,
            combatCat,
            workerCat,
            myQueenHPCat,
            queenThreatCat,
            hillThreatCat,
            foodDistCat,
            gameStageCat,
        )

        return category
