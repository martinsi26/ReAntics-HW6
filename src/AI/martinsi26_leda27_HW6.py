import random
import sys
import os
import pickle
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


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
        super(AIPlayer,self).__init__(inputPlayerId, "TD Learning Agent")
        #the coordinates of the agent's food and tunnel will be stored in these
        #variables (see getMove() below)
        self.myFood = None
        self.myTunnel = None
        
        # TD Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.lambda_trace = 0.7  # Eligibility trace decay
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        # State utilities mapping state categories to utility values
        self.stateUtilities = {}
        
        # Game tracking for TD learning
        self.stateHistory = []
        self.eligibilityTraces = {}
        
        # Statistics
        self.gamesPlayed = 0
        self.stateVisitCounts = {}  # Track how often each state is visited
        
        # Load saved utilities if they exist
        self.loadUtilities()
    
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
    # Uses TD Learning to select the best move based on learned state utilities
    #
    ##
    def getMove(self, currentState):
        # Get current state category
        currentCategory = self.stateToCategory(currentState)
        
        # Track state visit
        if currentCategory not in self.stateVisitCounts:
            self.stateVisitCounts[currentCategory] = 0
        self.stateVisitCounts[currentCategory] += 1
        
        # Get all legal moves
        legalMoves = listAllLegalMoves(currentState)
        
        if len(legalMoves) == 0:
            return Move(END, None, None)
        
        # Calculate intermediate reward for current state
        intermediateReward = self.calculateReward(currentState)
        
        # Add current state to history for TD learning
        self.stateHistory.append((currentCategory, intermediateReward))
        
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            # choose random move
            return random.choice(legalMoves)
        
        # choose move that leads to highest utility state
        bestMove = None
        bestUtility = float('-inf')
        
        for move in legalMoves:
            if move.moveType == END:
                # Ending turn is a neutral action, use current state utility
                utility = self.getStateUtility(currentCategory)
            else:
                # Predict next state after this move
                nextState = getNextState(currentState, move)
                nextCategory = self.stateToCategory(nextState)
                utility = self.getStateUtility(nextCategory)
            
            if utility > bestUtility:
                bestUtility = utility
                bestMove = move
        
        # If multiple moves have same utility, break ties randomly
        if bestMove is None:
            return random.choice(legalMoves)
        
        return bestMove
    
    ##
    #getAttack
    #
    # This agent attacks the first available enemy
    #
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        if len(enemyLocations) > 0:
            return enemyLocations[0]
        return None
        
    ##
    #calculateReward
    #
    # Calculates the reward for the current state
    # Returns +1 for win, -1 for loss, -0.01 for intermediate states
    #
    def calculateReward(self, state):
        winner = getWinner(state)
        if winner is not None:
            if winner == state.whoseTurn:
                return 1.0  # Win
            else:
                return -1.0  # Loss
        return -0.01
    
    ##
    #registerWin
    #
    # Updates state utilities using TD Learning with eligibility traces
    #
    def registerWin(self, hasWon):
        # Determine final reward
        if hasWon:
            finalReward = 1.0
        else:
            finalReward = -1.0
        
        # Update the last state in history with final reward
        if len(self.stateHistory) > 0:
            self.stateHistory[-1] = (self.stateHistory[-1][0], finalReward)
        
        # Apply TD Learning with eligibility traces
        self.updateUtilitiesWithEligibilityTraces(finalReward)
        
        # Reset for next game
        self.stateHistory = []
        self.eligibilityTraces = {}
        self.gamesPlayed += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Save utilities after each game
        self.saveUtilities()
    
    ##
    #updateUtilitiesWithEligibilityTraces
    #
    # Updates state utilities using TD(λ) with eligibility traces
    #
    def updateUtilitiesWithEligibilityTraces(self, finalReward):
        if len(self.stateHistory) == 0:
            return
        
        # Process states in forward order, updating eligibility traces as we go
        for i in range(len(self.stateHistory)):
            stateCat, reward = self.stateHistory[i]
            
            # Increment eligibility trace for current state
            if stateCat not in self.eligibilityTraces:
                self.eligibilityTraces[stateCat] = 0.0
            self.eligibilityTraces[stateCat] += 1.0
            
            # Calculate TD target
            if i == len(self.stateHistory) - 1:
                # Terminal state
                tdTarget = finalReward
            else:
                # Get the next state category and its utility
                nextStateCat, _ = self.stateHistory[i + 1]
                nextUtility = self.getStateUtility(nextStateCat)
                tdTarget = reward + self.gamma * nextUtility
            
            # TD error
            currentUtility = self.getStateUtility(stateCat)
            tdError = tdTarget - currentUtility
            
            # Update all state utilities using their eligibility traces
            for state in list(self.eligibilityTraces.keys()):
                eligibility = self.eligibilityTraces[state]
                update = self.alpha * tdError * eligibility
                
                # Update utility
                oldUtility = self.getStateUtility(state)
                self.stateUtilities[state] = oldUtility + update
                
                # Decay eligibility trace
                self.eligibilityTraces[state] *= self.gamma * self.lambda_trace
    
    ##
    #getStateUtility
    #
    # Returns the utility value for a given state category
    # Returns 0.0 if state hasn't been seen before
    #
    def getStateUtility(self, stateCategory):
        if stateCategory in self.stateUtilities:
            return self.stateUtilities[stateCategory]
        return 0.0  # Default utility for unseen states
    
    ##
    #binCalculation
    #
    # Helper method to bin a value into discrete categories
    #
    def binCalculation(self, value, bins):
        for i, b in enumerate(bins):
            if value < b:
                return i
        return len(bins)
    
    ##
    #stateToCategory
    #
    # Converts a game state into a discrete category for learning
    # This is critical for reducing the state space
    #
    def stateToCategory(self, state):
        # --- BINS ---
        foodBin = [0, 2, 5, 8, 10, 11]  # Absolute food bins
        foodDiffBin = [-6, -3, -1, 1, 3, 6]  # Relative food difference bins
        combatDiffBin = [-10, -3, 0, 3, 10]  # Combat power difference bins
        workerBin = [0, 1, 2, 3]  # Worker count bins
        queenHPBin = [0.3, 0.7]  # Queen HP bins (normalized)
        threatBin = [0, 1, 2]  # Threat bins
        foodDistBin = [4, 8, 12]  # Nearest food distance bins
        
        myInv = getCurrPlayerInventory(state)
        enemyId = 1 - state.whoseTurn
        enemyInv = state.inventories[enemyId]
        
        # --- Food ---
        myFood = myInv.foodCount
        enemyFood = enemyInv.foodCount
        myFoodCat = self.binCalculation(myFood, foodBin)
        enemyFoodCat = self.binCalculation(enemyFood, foodBin)
        foodDiff = myFood - enemyFood
        foodDiffCat = self.binCalculation(foodDiff, foodDiffBin)
        
        # --- Combat Power ---
        myCombatPower = self.calculateCombatPower(state, state.whoseTurn)
        enemyCombatPower = self.calculateCombatPower(state, enemyId)
        combatDiff = myCombatPower - enemyCombatPower
        combatCat = self.binCalculation(combatDiff, combatDiffBin)
        
        # --- Workers ---
        myWorkers = getAntList(state, state.whoseTurn, (WORKER,))
        numWorkers = len(myWorkers)
        workerCat = self.binCalculation(numWorkers, workerBin)
        
        # --- Queen HP ---
        myQueen = myInv.getQueen()
        if myQueen:
            maxQueenHP = UNIT_STATS[QUEEN][HEALTH]
            myQueenHPNorm = myQueen.health / maxQueenHP if maxQueenHP > 0 else 0.0
            myQueenHPCat = self.binCalculation(myQueenHPNorm, queenHPBin)
        else:
            myQueenHPCat = 0
        
        enemyQueen = enemyInv.getQueen()
        if enemyQueen:
            enemyQueenHPNorm = enemyQueen.health / maxQueenHP if maxQueenHP > 0 else 0.0
            enemyQueenHPCat = self.binCalculation(enemyQueenHPNorm, queenHPBin)
        else:
            enemyQueenHPCat = 0
        
        # --- Threats ---
        queenThreat = self.calculateThreat(state, myQueen.coords if myQueen else None, enemyId)
        queenThreatCat = self.binCalculation(queenThreat, threatBin)
        
        hillThreat = self.calculateThreat(state, myInv.getAnthill().coords, enemyId)
        hillThreatCat = self.binCalculation(hillThreat, threatBin)
        
        # --- Nearest Food Distance ---
        nearestFoodDist = self.getNearestFoodDistance(state)
        foodDistCat = self.binCalculation(nearestFoodDist, foodDistBin)
        
        # --- Return tuple ---
        category = (
            myFoodCat,
            enemyFoodCat,
            foodDiffCat,
            combatCat,
            workerCat,
            myQueenHPCat,
            enemyQueenHPCat,
            queenThreatCat,
            hillThreatCat,
            foodDistCat
        )
        
        return category
    
    ##
    #calculateCombatPower
    #
    # Calculates total combat power of a player's ants
    #
    def calculateCombatPower(self, state, playerId):
        ants = getAntList(state, playerId)
        totalPower = 0
        for ant in ants:
            totalPower += UNIT_STATS[ant.type][ATTACK] * ant.health
        return totalPower
    
    ##
    #calculateThreat
    #
    # Counts enemy ants within attack range of a location
    #
    def calculateThreat(self, state, location, enemyId):
        if location is None:
            return 0
        
        threatCount = 0
        enemyAnts = getAntList(state, enemyId)
        
        for ant in enemyAnts:
            dist = approxDist(ant.coords, location)
            if dist <= UNIT_STATS[ant.type][RANGE]:
                threatCount += 1
        
        return threatCount
    
    ##
    #getNearestFoodDistance
    #
    # Returns the distance to the nearest food from any worker
    #
    def getNearestFoodDistance(self, state):
        myWorkers = getAntList(state, state.whoseTurn, (WORKER,))
        if len(myWorkers) == 0:
            return 20  # Large distance if no workers
        
        foods = getConstrList(state, None, (FOOD,))
        if len(foods) == 0:
            return 20
        
        minDist = 20
        for worker in myWorkers:
            for food in foods:
                dist = approxDist(worker.coords, food.coords)
                if dist < minDist:
                    minDist = dist
        
        return minDist
    
    ##
    #saveUtilities
    #
    # Saves the current state utilities to a file
    #
    def saveUtilities(self):
        filename = "martinsi26_leda27_weights.pkl"
        try:
            with open(filename, 'wb') as f:
                data = {
                    'utilities': self.stateUtilities,
                    'visitCounts': self.stateVisitCounts,
                    'gamesPlayed': self.gamesPlayed,
                    'epsilon': self.epsilon
                }
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving utilities: {e}")
    
    ##
    #loadUtilities
    #
    # Loads state utilities from a file if it exists
    #
    def loadUtilities(self):
        filename = "martinsi26_leda27_weights.pkl"
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.stateUtilities = data.get('utilities', {})
                    self.stateVisitCounts = data.get('visitCounts', {})
                    self.gamesPlayed = data.get('gamesPlayed', 0)
                    self.epsilon = data.get('epsilon', self.epsilon)
                    print(f"Loaded {len(self.stateUtilities)} state utilities from {filename}")
        except Exception as e:
            print(f"Error loading utilities: {e}")
            # Initialize empty if load fails
            self.stateUtilities = {}
            self.stateVisitCounts = {}
