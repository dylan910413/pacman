from pacman import SCARED_TIME, GameState
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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        def value(gameState,agentIndex,currentdepth) :
            if gameState.isWin() or gameState.isLose() or currentdepth == self.depth :
                return (self.evaluationFunction(gameState),0)
            if agentIndex == 0 :
                return max_value(gameState,agentIndex,currentdepth)
            else :
                return min_value(gameState,agentIndex,currentdepth)
        def max_value(gameState,agentIndex,currentdepth) :
            v = -9999999999
            nextagentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextagentIndex == 0 :
                currentdepth+=1
            for action in gameState.getLegalActions(agentIndex) :
                nextstate = gameState.getNextState(agentIndex, action)
                nextV = value(nextstate,nextagentIndex,currentdepth)[0]
                if nextV > v :
                    v = nextV
                    a = action
            return (v,a)
        def min_value(gameState,agentIndex,currentdepth) :
            v = +9999999999
            nextagentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextagentIndex == 0 :
                currentdepth+=1
            for action in gameState.getLegalActions(agentIndex) :
                nextstate = gameState.getNextState(agentIndex, action)
                nextV = value(nextstate,nextagentIndex,currentdepth)[0]
                if nextV < v :
                    v = nextV
                    a = action
            return (v,a)
        return value(gameState,0,0)[1]
            
        #raise NotImplementedError("To be implemented")

        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        def value(gameState,agentIndex,currentdepth,alpha,beta) :
            if gameState.isWin() or gameState.isLose() or currentdepth == self.depth :
                return (self.evaluationFunction(gameState),0)
            if agentIndex == 0 :
                return max_value(gameState,agentIndex,currentdepth,alpha,beta)
            else :
                return min_value(gameState,agentIndex,currentdepth,alpha,beta)
        def max_value(gameState,agentIndex,currentdepth,alpha,beta) :
            v = -9999999999
            nextagentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextagentIndex == 0 :
                currentdepth+=1
            for action in gameState.getLegalActions(agentIndex) :
                nextstate = gameState.getNextState(agentIndex, action)
                nextV = value(nextstate,nextagentIndex,currentdepth,alpha,beta)[0]
                if nextV > v :
                    v = nextV
                    a = action
                    if v > beta :
                        return (v,a)
                    alpha = max(alpha,v)
            return (v,a)
        def min_value(gameState,agentIndex,currentdepth,alpha,beta) :
            v = +9999999999
            nextagentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextagentIndex == 0 :
                currentdepth+=1
            for action in gameState.getLegalActions(agentIndex) :
                nextstate = gameState.getNextState(agentIndex, action)
                nextV = value(nextstate,nextagentIndex,currentdepth,alpha,beta)[0]
                if nextV < v :
                    v = nextV
                    a = action
                    if v < alpha :
                        return (v,a)
                    beta = min(beta,v)
            return (v,a)
        return value(gameState,0,0,-9999999999,9999999999)[1]
        raise NotImplementedError("To be implemented")
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        def value(gameState,agentIndex,currentdepth) :
            if gameState.isWin() or gameState.isLose() or currentdepth == self.depth :
                return (self.evaluationFunction(gameState),0)
            if agentIndex == 0 :
                return max_value(gameState,agentIndex,currentdepth)
            else :
                return ave_value(gameState,agentIndex,currentdepth)
        def max_value(gameState,agentIndex,currentdepth) :
            v = -9999999999
            nextagentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextagentIndex == 0 :
                currentdepth+=1
            for action in gameState.getLegalActions(agentIndex) :
                nextstate = gameState.getNextState(agentIndex, action)
                nextV = value(nextstate,nextagentIndex,currentdepth)[0]
                if nextV > v :
                    v = nextV
                    a = action
            return (v,a)
        def ave_value(gameState,agentIndex,currentdepth) :
            nextagentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextagentIndex == 0 :
                currentdepth+=1
            numofact = 0
            sum = 0
            for action in gameState.getLegalActions(agentIndex) :
                numofact += 1
                nextstate = gameState.getNextState(agentIndex, action)
                nextV = value(nextstate,nextagentIndex,currentdepth)[0]
                sum += nextV
            ave = sum/numofact
            return (ave,0)
        return value(gameState,0,0)[1]
        raise NotImplementedError("To be implemented")
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    score = currentGameState.getScore()
    newpos = currentGameState.getPacmanPosition()
    foodposes = currentGameState.getCapsules()
    distancesToFoodList = [manhattanDistance(newpos, foodpos) for foodpos in foodposes]
    if len(distancesToFoodList) > 0:
        score += 50 / min(distancesToFoodList)
    else:
        score += 50
    
    return score
    raise NotImplementedError("To be implemented")
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
