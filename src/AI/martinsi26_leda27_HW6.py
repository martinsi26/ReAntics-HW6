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
        #the coordinates of the agent's food and tunnel will be stored in these
        #variables (see getMove() below)
        self.myFood = None
        self.myTunnel = None
    
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
        pass
                              
    
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
        #method templaste, not implemented
        pass

    def binCalculation(value, bin):
        for i, b, in enumerate(bin):
            if value < b:
                return i
        return len(bin)

    def stateToCategory(self, state):
        # --- BINS ---
        foodBin = [0, 2, 5, 8, 10, 11] # Absolute food bins (for AI and enemy)
        foodDiffBin = [-6, -3, -1, 1, 3, 6] # Relative food difference bins
        combatDiffBin = [-10, -3, 0, 3, 10] # Combat power difference bins
        workerBin = [0, 1, 2] # Worker count bins
        queenHPBin = [0.3, 0.7] # Queen HP bins (0–1 normalized)
        threatBin = [0, 2] # Threat bins (queen/hill)
        foodDistBin = [4, 8, 16] # Nearest food distance bins
        
        myInv = getCurrPlayerInventory(state)
        enemyInv = getEnemyInv(state)

        # --- Food ---
        myFood = myInv.foodCount
        enemyFood = enemyInv.foodCount
        myFoodCat = self.binCalculation(myFood, foodBin)
        enemyFoodCat = self.binCalculation(enemyFood, foodBin)
        foodDiff = myFood - enemyFood
        foodDiffCat = self.binCalculation(foodDiff, foodDiffBin)

        # --- Combat ---
        
        # --- Workers ---
        myWorkers = getAntList(state, state.whoseTurn, (WORKER,))
        numWorkers = len(myWorkers)
        workerCat = self.binCalculation(numWorkers, workerBin)

        # --- Queen HP ---
        myQueen = myInv.getQueen()
        myQueenHP = myQueen.health
        myQueenHPCat = self.binCalculation(myQueenHP, queenHPBin)

        # --- Threats ---
        queenThreatCat = self.binCalculation(_, threatBin)

        # --- Path ---

        # --- Nearest Food Distance ---
        


        # --- Return tuple ---
        category = (
            myFoodCat,
            enemyFoodCat,
            foodDiffCat,
            combatCat,
            workerCat,
            myQueenHPCat,
            enemy_qhp_bin,
            qthreat_bin,
            hthreat_bin,
            path_bin,
            food_dist_bin
        )

        return category
