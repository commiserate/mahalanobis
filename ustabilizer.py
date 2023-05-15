from copy import deepcopy, copy
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

NUMBERED_STATES = [
    (1,1), (2,2),
    (2,0), (2,3),
    (3,0), (3,3),
    (1,4), (4,2)
]
INVALID_STATES = [
    (0,1), (0,2),
    (2,1), (0,3),
    (3,1), (0,4),
    (4,1), (1,3),
    (4,3)
]
STATE_EXITED = (-1, -1)

ACT_LEFT = '←'
ACT_RIGHT = '→'
ACT_UP = '↑'
ACT_DOWN = '↓'
ACT_EXIT = 'X'
ALL_ACTS = [
    ACT_LEFT,
    ACT_RIGHT,
    ACT_UP,
    ACT_DOWN,
    ACT_EXIT
]

def R(state, act):
    if act != ACT_EXIT:
        return 0
    r = [
            [0, None, None, None, None],
            [0,    1,    0, None,   -1],
            [4, None,    8,    1,    0],
            [3, None,    0,    2,    0],
            [0, None,    2, None,    0],
            [0,    0,    0,    0,    0],
    ]
    i,j = state
    # print(f'\tR({state} =? {r[i][j]})')
    return r[i][j]


def distToClosestNS(state):
    i,j = state
    dists = []
    for ns in NUMBERED_STATES:
        dist = manhattan_distances([state], [ns])
        dists.append( (dist, ns) )
    
    minDistAndNS = min(dists)
    minDist = minDistAndNS[0]
    return minDist


def F1(startState, nextState):
    if nextState in NUMBERED_STATES:
        return 0
    if distToClosestNS(nextState) < distToClosestNS(startState):
        return 8
    else:
        return 0


class ValueGrid(object):
    def __init__(self, uGrid, policy=[], time=1):
        assert type(uGrid) == list
        self.grid = uGrid
        self.policy = policy
        if self.policy == []:
            self.policy = [
                ['█' for _ in range(len(self.grid[0])) ] for _ in range(len(self.grid))
            ]
        self.time = time

    def __str__(self):
        return str(self.grid)+'\n'+str(self.policy)

    def __eq__(self, other):
        assert type(other) == ValueGrid
        
        selfgrid_arr = np.array(self.grid, dtype=float)
        othergrid_arr = np.array(other.grid, dtype=float)

        # print(selfgrid_arr)
        sameGrid = np.allclose(selfgrid_arr, othergrid_arr, equal_nan=True)
        # sameGrid = self.grid == other.grid

        samePolicy = self.policy == other.policy
        return samePolicy
        # return sameGrid and samePolicy

    def getStateAfterAct(self, state, act):
        if type(state) == tuple:
            assert len(state) == 2
            assert type(state[0]) == int
            assert type(state[1]) == int
            i,j = state
        elif type(state) == int:
            i = state

        if act == ACT_EXIT:
            return STATE_EXITED

        if act == ACT_RIGHT:
            j += 1
        if act == ACT_DOWN:
            i += 1
        if act == ACT_LEFT:
            j -= 1
        if act == ACT_UP:
            i -= 1
        # print(i,j, end='')
        if i >= len(self.grid):
            i = len(self.grid) - 1
        if i < 0:
            i = 0

        if j >= len(self.grid[0]):
            j = len(self.grid[0]) - 1
        if j < 0:
            j = 0
        # print(' ->',i,j)
        return i,j
    
    def getExpectedValue(self, state, intendedAct):
        # print(f'-- getExpectedValue {state} {intendedAct}')
        i,j = self.getStateAfterAct(state, intendedAct)
        # print('state after act:',i,j)
        if (i,j) == STATE_EXITED:
            return R(state, ACT_EXIT)
        return self.grid[i][j]

    def setGrid(self, state, value):
        self.grid[state[0]][state[1]] = value
    
    def setPolicy(self, state, value):
        # print('-'*15,'VG.setPolicy')
        # print(state, value)
        # [print(row) for row in self.policy]
        self.policy[state[0]][state[1]] = value
# end class

def getScoreMaximizingAct(state, thisVG):
    validActs = copy(ALL_ACTS)
    i,j = state
    
    if j==0:
        validActs.remove(ACT_LEFT)
    if i==0:
        validActs.remove(ACT_UP)
    if i==5:
        validActs.remove(ACT_DOWN)
    if j==4:
        validActs.remove(ACT_RIGHT)
    if state not in NUMBERED_STATES:
        validActs.remove(ACT_EXIT)
    # print('--- getScoreMaximizingAct')
    # print('validActs', validActs)
    values = [(act, thisVG.getExpectedValue(state, act)) for act in validActs]
    values = [(act,value) for act,value in values if value != None] # filter out moves into blocked "Wall" spaces
    if i==0 and j in [2,3]:
        print(f'{state} act,score pairs',values)
    values.sort(key=lambda actScorePair:actScorePair[1], reverse=True)
    act, value = values[0]
    return value, act

def printOptimalPolicy(gamma, DEBUG=False):
    # U_zero = [
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0]
    # ]
    U_zero = [
        [0, None, None, None, None],
        [0,    0,    0, None,    0],
        [0, None,    0,    0,    0],
        [0, None,    0,    0,    0],
        [0, None,    0, None,    0],
        [0,    0,    0,    0,    0],
        # [0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0],
    ]
    prevU = ValueGrid(U_zero, policy=[], time=0)
    policyUpdateCount = -1
    while True:
        # input()
        policyUpdateCount += 1
        
        newU = ValueGrid( deepcopy(prevU.grid), policy=[], time=prevU.time+1 )
        
        # for each cell in new U grid
        if DEBUG:
            print('Update each cell')
        for i in range(len(U_zero)):
            for j in range(len(U_zero[0])):
                if (i,j) in INVALID_STATES:
                    continue
                state = (i,j)
                maxAsum, act = getScoreMaximizingAct(state, prevU)
                stateAfterAct = newU.getStateAfterAct(state, act)
                prodVal = R(state, act) + F1(state, stateAfterAct) + (gamma * maxAsum)
                # prodVal = R(state, act) + (gamma * maxAsum)
                if DEBUG:
                    print()
                    print('State', state)
                    print('Score/Act',maxAsum,act)
                    print('new val for next policy iteration',prodVal)
                newU.setGrid(state, prodVal)
                newU.setPolicy(state, act)

        stabilized = newU == prevU

        prevU = ValueGrid(newU.grid, newU.policy)

        # print(policyUpdateCount)
        rowLen = len("['↓', '█', '█', '█', '█']")
        print()
        print('-'*rowLen)
        print()
        if policyUpdateCount > 90:
            for row in prevU.grid:
                print([num for num in row])
        for row in prevU.policy:
            print(row)

        if stabilized or policyUpdateCount == 100:
            print('-'*50, 'COMPLETE')
            print(policyUpdateCount, 'iterations')
            for row in prevU.grid:
                print([num for num in row])
            for row in prevU.policy:
                print(row)
            # print('stabilized')
            # print(gamma, prevU.goesR())
            
            break

if __name__ == '__main__':
    gamma = 0.5
    print(printOptimalPolicy(gamma, DEBUG=False))

# doesThisGammaGoR(gamma)
# gammas = np.linspace(0.1221296875, 0.12212978515625, num=9)
# print(gammas)
# for g in gammas:
    # goR = doesThisGammaGoR(g)
    # print(goR,'\t', round(g,6), g)


# print(prevU.policy)