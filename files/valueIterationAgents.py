# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

        

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        iteration = 0
        # running iterations to get values of each iteration and finding the best
        while iteration < self.iterations:
            # initialize counter into values
            values = self.values.copy()
            # for each state in mdp check if terminal then go thru each iteration to get values
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                # get best actions from each state based on values
                action = self.computeActionFromValues(state)
                # compute q values
                qVal = self.computeQValueFromValues(state, action)
                # add that to our counter
                values[state] = qVal
            self.values = values
            # add on to show iteration has been counted for
            iteration += 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # go through the actions to get the values to compute q value
        nextActions = self.mdp.getTransitionStatesAndProbs(state,action)
        # initalize value 
        value = 0
        # go over all possible actions and compute the q value
        for actions in nextActions:
            value += actions[1] * (self.mdp.getReward(state, action,actions[0])+self.discount*self.values[actions[0]])
        return value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # check if it is terminal
        if self.mdp.isTerminal(state):
            return None
        
        # get possible actions
        actions = self.mdp.getPossibleActions(state)

        # initialize q value for actions values
        qValue =  util.Counter()

        for action in actions:
            qValue[action] = self.computeQValueFromValues(state, action)
        
        # return the best action
        return qValue.argMax()



        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # compute predecessors of all states
        self.predecessors = util.Counter()
        # initialize an empty priority queue
        self.queue = util.PriorityQueue()

        # for each non terminal state s, do 
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                self.predecessors[s] = set()

        # find absolute value of difference betweeen current value of s and best q value of s
        # assign this to diff
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                values = self.computeActionFromValues(s)
                # get best q val from values
                bestQVal = self.computeQValueFromValues(s, values)
                currVal = self.values[s]
                diff = abs(currVal - bestQVal)

                # push s into the priority queue with priority -diff 
                self.queue.push(s, -diff)   

        # for iteration 
        for iter in range(self.iterations):

            #if priority queue is empty then terminate
            if self.queue.isEmpty():
                break

        # pop a state s off the priority queue
        s = self.queue.pop()

        # if not in terminal state then update the values of s
        if not self.mdp.isTerminal(s):
            action = self.computeActionFromValues(s)
            self.values[s] = self.computeQValueFromValues(s, values)

        # for each predecessor p of s
        for p in self.predecessors[s]:
        # Find the absolute value of the difference between the current value of p in
        # self.values and the highest Q-value across all possible actions from p 
        # (this represents what the value should be); call this number diff. 
           
            actions = self.computeActionFromValues(p)
            QvalBest = self.computeQValueFromValues(p, actions)
            # look over all actions to find best q val
            currentVal = self.values[p]
            # use q val and current value to get absolute value of their difference
            diff = abs(currentVal - QvalBest)
            # if diff > theta, push p onto priority queue with priority -diff
            if diff > self.theta:
                self.queue.update(p, -diff)


