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
        #Just put in my previous method for starting the game, can change to better strategy
        self.myFood = None
        self.myTunnel = None

        if currentState.phase == SETUP_PHASE_1:
            return [
                (1, 1), (8, 1),  # Anthill and hive
                #Make a Grass wall
                (0, 3), (1, 3), (2, 3), (3, 3),  #Grass 
                (4, 3), (5, 3), (6, 3), #Grass
                (8, 3), (9, 3) # Grass
            ]
        #Placing the enemies food (In the corners/randomly far away from their anthill)
        elif currentState.phase == SETUP_PHASE_2:
            #The places the method will choose and append to return
            foodSpots = []
            #Corner coordinates
            corners = [(0, 9), (0, 6), (9, 6), (9, 9)]

            #Go through corners, make sure its legal and add to the return list
            for coord in corners:
                if legalCoord(coord) and getConstrAt(currentState, coord) is None:
                    foodSpots.append(coord)
                #If you have both spots, break and go to return
                if len(foodSpots) == 2:
                    break
            #If one or more of the corners are covered pick a random spot
            while len(foodSpots) < 2:
                coord = (random.randint(0, 9), random.randint(6, 9))
                if legalCoord(coord) and getConstrAt(currentState, coord) is None and coord not in foodSpots:
                    foodSpots.append(coord)

            #Return final list of enemy food placement
            return foodSpots

        return None
    
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
        legalMoves = self.filterLegalMoves(currentState, legalMoves)  # <-- filter workers
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
    
    # --- Hard cap for workers ---
    MAX_WORKERS = 2 
    # --- Hard cap for combat units ---
    MAX_COMBAT_UNITS = 2

    def filterLegalMoves(self, state, moves):
        """
        Remove build moves if max units are reached.
        """
        # Count current workers
        myWorkers = getAntList(state, state.whoseTurn, (WORKER,))
        # Count current combat units
        myCombat = getAntList(state, state.whoseTurn, (SOLDIER, R_SOLDIER, DRONE))

        filtered = []

        for m in moves:
            if hasattr(m, 'buildType'):
                if m.buildType == WORKER and len(myWorkers) >= self.MAX_WORKERS:
                    continue  # skip building extra workers
                if m.buildType in (SOLDIER, R_SOLDIER, DRONE) and len(myCombat) >= self.MAX_COMBAT_UNITS:
                    continue  # skip building extra combat units
            filtered.append(m)

        # If filtering removes everything, fallback to all moves
        return filtered if filtered else moves




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
        nearestFoodDistBin = [0, 1, 2, 4, 6] # Nearest food distance bins
        nearestDropoffDistBin = [0, 1, 2, 4, 6] # Nearest drop off location distance bins
        attackDistBin = [0, 1, 2, 3, 5, 7, 10]  # distance bins for attacking ants to enemy queen
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
                if a.coords and myQueen.coords and stepsToReach(state, a.coords, myQueen.coords) <= 2:
                    queenThreat = 1
                if a.coords and hill.coords and stepsToReach(state, a.coords, hill.coords) <= 2:
                    hillThreat = 1

        queenThreatCat = getBin(queenThreat, threatBin)
        hillThreatCat = getBin(hillThreat, threatBin)

        # --- Worker distances for food efficiency ---
        foodLocations = getConstrList(state, None, (FOOD,))
        tunnel = myInv.getTunnels()

        bestDistToFood = 999
        bestDistToDropoff = 999

        for w in myWorkers:
            if not w.coords:
                continue
            if not w.carrying:
                # Worker going to nearest food
                for food in foodLocations:
                    d = stepsToReach(state, w.coords, food.coords)
                    bestDistToFood = min(bestDistToFood, d)
            else:
                # Worker carrying food → nearest drop-off (anthill or tunnel)
                dHill = stepsToReach(state, w.coords, hill.coords)
                dTunnel = stepsToReach(state, w.coords, tunnel[0].coords) if tunnel else 999
                bestDistToDropoff = min(bestDistToDropoff, dHill, dTunnel)

        # Ensure distances default to max if no workers
        nearestFoodDistCat = getBin(bestDistToFood, nearestFoodDistBin)
        nearestDropoffDistCat = getBin(bestDistToDropoff, nearestDropoffDistBin)

        # ---------- GAME STAGE ----------
        myProgress = myInv.foodCount / 11.0
        enemyProgress = enemyInv.foodCount / 11.0
        
        gameProgress = max(myProgress, enemyProgress)
        gameStageCat = getBin(gameProgress, gameStageBins)

        # --- Attack distance to enemy queen ---
        # --- Attack distance to enemy queen ---
        enemyQueen = enemyInv.getQueen()
        attackingAnts = getAntList(state, state.whoseTurn, (SOLDIER, R_SOLDIER, DRONE))

        minAttackDist = 999  # large number initially
        if enemyQueen is not None and enemyQueen.coords is not None:
            for a in attackingAnts:
                if a.coords:
                    d = stepsToReach(state, a.coords, enemyQueen.coords)
                    if d < minAttackDist:
                        minAttackDist = d
        else:
            # No enemy queen → keep minAttackDist high
            minAttackDist = 999

        # Bin the distance
        attackDistCat = getBin(minAttackDist, attackDistBin)

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
            nearestFoodDistCat,
            nearestDropoffDistCat,
            attackDistCat,
            gameStageCat,
        )

        return category
