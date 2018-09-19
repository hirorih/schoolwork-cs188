# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_list = newFood.asList()

        nearest_ghost = min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates])


        if nearest_ghost:
            ghost_weight = -10 / nearest_ghost
        else:
            ghost_weight = -1000

        if food_list:
            nearest_food = min([manhattanDistance(newPos, food) for food in food_list])
        else:
            nearest_food = 0

        food_weight = -5 * nearest_food
        food_left = -100 * len(food_list)

        return food_left + food_weight + ghost_weight

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"

        minimax_value, minimax_action = self.minimax_value(gameState, self.index, 0)
        return minimax_action

    def minimax_value(self, gameState, agentIndex, searchDepth):

        agents_num = gameState.getNumAgents()

        if agentIndex % agents_num == 0 and searchDepth == self.depth:
            # terminal check returns eval and action=none
            return self.evaluationFunction(gameState), None

        # when the agent is packman
        if agentIndex % agents_num == 0:
            return self.max_value(gameState, agentIndex % agents_num, searchDepth)
        # when the agent is ghost
        return self.min_value(gameState, agentIndex % agents_num, searchDepth)

    def max_value(self, gameState, agentIndex, searchDepth):

        legalMoves = gameState.getLegalActions(agentIndex)
        successor_states = [(gameState.generateSuccessor(agentIndex, action), action) for action in legalMoves]
        if len(successor_states) == 0:
            return self.evaluationFunction(gameState), None

        v = -float("inf")
        a = None
        for successor_state, action in successor_states:
            next_value, next_action = self.minimax_value(successor_state, agentIndex + 1, searchDepth + 1)
            if next_value > v:
                v = next_value
                a = action
        return v, a

    def min_value(self, gameState, agentIndex, searchDepth):

        legalMoves = gameState.getLegalActions(agentIndex)
        successor_states = [(gameState.generateSuccessor(agentIndex, action), action) for action in legalMoves]
        if len(successor_states) == 0:
            return self.evaluationFunction(gameState), None

        v = float("inf")
        a = None
        for successor_state, action in successor_states:
            next_value, next_action = self.minimax_value(successor_state, agentIndex + 1, searchDepth)
            if next_value < v:
                v = next_value
                a = action
        return v, a

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        minimax_value, minimax_action = self.minimax_alpha_beta(gameState, self.index, 0, -float("inf"), float("inf"))
        return minimax_action

    def minimax_alpha_beta(self, gameState, agentIndex, searchDepth, alpha, beta):

        agents_num = gameState.getNumAgents()

        if agentIndex % agents_num == 0 and searchDepth == self.depth:
            # terminal check returns eval and action=none
            return self.evaluationFunction(gameState), None

        # when the agent is packman
        if agentIndex % agents_num == 0:
            return self.max_alpha_beta(gameState, agentIndex % agents_num, searchDepth, alpha, beta)
        # when the agent is ghost
        return self.min_alpha_beta(gameState, agentIndex % agents_num, searchDepth, alpha, beta)

    def max_alpha_beta(self, gameState, agentIndex, searchDepth, alpha, beta):

        legalMoves = gameState.getLegalActions(agentIndex)

        if len(legalMoves) == 0:
            return self.evaluationFunction(gameState), None

        v = -float("inf")
        a = None
        for action in legalMoves:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            next_value, next_action = self.minimax_alpha_beta(successor_state, agentIndex + 1, searchDepth + 1, alpha, beta)
            if next_value > v:
                v = next_value
                a = action
            if v > beta:
                return v, a
            alpha = max(alpha, v)
        return v, a

    def min_alpha_beta(self, gameState, agentIndex, searchDepth, alpha, beta):

        legalMoves = gameState.getLegalActions(agentIndex)

        if len(legalMoves) == 0:
            return self.evaluationFunction(gameState), None

        v = float("inf")
        a = None
        for action in legalMoves:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            next_value, next_action = self.minimax_alpha_beta(successor_state, agentIndex + 1, searchDepth, alpha, beta)
            if next_value < v:
                v = next_value
                a = action
            if v < alpha:
                return v, a
            beta = min(beta, v)
        return v, a


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        expectimax_value, expectimax_action = self.expectimax(gameState, self.index, 0)
        return expectimax_action

    def expectimax(self, gameState, agentIndex, searchDepth):
        agents_num = gameState.getNumAgents()

        if agentIndex % agents_num == 0 and searchDepth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex % agents_num == 0:
            return self.max_value(gameState, agentIndex % agents_num, searchDepth)
        return self.expected_value(gameState, agentIndex % agents_num, searchDepth)

    def max_value(self, gameState, agentIndex, searchDepth):

        legalMoves = gameState.getLegalActions(agentIndex)
        successor_states = [(gameState.generateSuccessor(agentIndex, action), action) for action in legalMoves]
        if len(successor_states) == 0:
            return self.evaluationFunction(gameState), None

        v = -float("inf")
        a = None
        for successor_state, action in successor_states:
            next_value, next_action = self.expectimax(successor_state, agentIndex + 1, searchDepth + 1)
            if next_value > v:
                v = next_value
                a = action
        return v, a

    def expected_value(self, gameState, agentIndex, searchDepth):

        legalMoves = gameState.getLegalActions(agentIndex)
        successor_states = [(gameState.generateSuccessor(agentIndex, action), action) for action in legalMoves]
        if len(successor_states) == 0:
            return self.evaluationFunction(gameState), None

        v = 0
        a = None
        n = 0
        for successor_state, action in successor_states:
            next_value, next_action = self.expectimax(successor_state, agentIndex + 1, searchDepth)
            v += next_value
            n += 1
        return v/n, a

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFood_lst = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    ghost_distance = min(manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates)
    food_distance = min(manhattanDistance(newPos, food) for food in newFood_lst) if newFood_lst else 0

    food_left = -len(newFood_lst)

    ghost_weight = -3 / (ghost_distance + 1) if min(newScaredTimes) == 0 else 1 / (ghost_distance + 1)

    power_pellets_weight = min(newScaredTimes) * 2

    food_weight = 1 / (food_distance + 1)

    score_weight = currentGameState.getScore()

    return food_left + ghost_distance + food_weight + power_pellets_weight + score_weight


# Abbreviation
better = betterEvaluationFunction
