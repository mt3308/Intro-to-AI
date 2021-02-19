
from __future__ import division
from __future__ import print_function
from collections import deque

import sys
import math
import resource
import time
import queue as Q

class Frontier:
    def __init__(self, initial_state):
        
        self.q = deque()
        self.q.append(initial_state)
        self.configSet = set()
        self.configSet.add(tuple(initial_state.config))
        self.len = len(self.q)
        self.heap = []
        self.heap.append(initial_state)

    
    def enqueue(self, x):
        self.q.append(x)
        self.configSet.add(tuple(x.config))
        
    def dequeue(self):
        x = self.q.popleft()
        self.configSet.remove(tuple(x.config))
        return x
    
    def push(self, x):
        self.q.append(x)
        self.configSet.add(tuple(x.config))
    
    def pop(self):
        x = self.q.pop()
        self.configSet.remove(tuple(x.config))
        return x

    def insert(self, x):
        self.heap.append(x)
        self.configSet.add(tuple(x.config))
    
    def deleteMin(self):
        x = self.heap[0]
        for i in self.heap:
            if calculate_total_cost(i) < calculate_total_cost(x):
                x = i
            elif calculate_total_cost(i) == calculate_total_cost(x):
                if self.calc_action(i) < self.calc_action(x):
                    x = i
        
        self.heap.remove(x)
        self.configSet.remove(tuple(x.config))
        return x
    
    def decreaseKey(self, x):
        for i in self.heap:
            if i.config == x.config:
                self.heap.remove(i)
        self.heap.append(x)
        
    def isEmpty(self):
        if len(self.q) == 0:
            return 1
        elif len(self.heap) == 0:
            return 1
        else:
            return 0
    
    def calc_action(self,x):
        value = 0
        if x.action == "Down":
            value = 1
        elif x.action == "Left":
            value = 2
        elif x.action == "Right":
            value = 3
        return value
            
            
#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        up_index = self.blank_index - self.n
        if up_index < 0:
            return None
        else:
            new_config = self.config.copy()
            new_config[self.blank_index] = new_config[up_index]
            new_config[up_index] = 0
        
            return PuzzleState(new_config, self.n, self, "Up", cost = self.cost + 1)
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        down_index = self.blank_index + self.n
        if down_index > self.n ** 2 - 1:
            return None
        else:
            new_config = self.config.copy()
            new_config[self.blank_index] = new_config[down_index]
            new_config[down_index] = 0
        
            return PuzzleState(new_config, self.n, self, "Down", cost = self.cost + 1)
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        left_index = self.blank_index - 1
        if left_index < 0 or left_index % self.n == self.n - 1:
            return None
        else:
            new_config = self.config.copy()
            new_config[self.blank_index] = new_config[left_index]
            new_config[left_index] = 0
        
            return PuzzleState(new_config, self.n, self, "Left", cost = self.cost + 1)

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        right_index = self.blank_index + 1
        if right_index > self.n ** 2 - 1 or right_index % self.n == 0:
            return None
        else:
            new_config = self.config.copy()
            new_config[self.blank_index] = new_config[right_index]
            new_config[right_index] = 0
            
            return PuzzleState(new_config, self.n, self, "Right", cost = self.cost + 1)
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(initial_state, final_state, n, m, c, r, s):
    ### Student Code Goes here
    f= open("output.txt", "w")
    
    path_to_goal = []
    search_depth = 0
    
    current = final_state
    while (current != initial_state):
        path_to_goal.insert(0, current.action)
        current = current.parent
        search_depth += 1
    
    f.write("path_to_goal: {}\ncost_of_path: {}\nnodes_expanded: {}\nsearch_depth: {}\nmax_search_depth: {}\nrunning_time: {}\nmax_Ram_usage:{}".format(path_to_goal, c, n, search_depth, m, round(r, 8), round(s, 8)))
    
    
    f.close()
    
    pass

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###

    frontier = Frontier(initial_state)
    explored = set()

    
    n=0 #nodes_expanded
    m=0
    s  = time.time()
    s_m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    while (frontier.isEmpty() == 0):
        state = frontier.dequeue()
        if (state.cost > m):
            m = state.cost
        explored.add(tuple(state.config))
        
        if test_goal(state):
            l = frontier.len
            r = time.time() - s
            sp = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - s_m
            writeOutput(initial_state, state, n, state.cost + l, state.cost, r, sp)
            return 1
        
        neighbors = state.expand()
        
        for neighbor in neighbors:
            if (tuple(neighbor.config) not in frontier.configSet and tuple(neighbor.config) not in explored):
                frontier.enqueue(neighbor)
        
        n += 1
    
    
    r = time.time()
        
    return 0
    

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    frontier = Frontier(initial_state)
    explored = set()
    
    n = 0 #nodes_expanded
    m = 0
    s  = time.time()
    s_m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    while (frontier.isEmpty() == 0):
        state = frontier.pop()
        if (state.cost > m):
            m = state.cost
            
        explored.add(tuple(state.config))
        
        if test_goal(state):
            r = time.time() - s
            sp = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - s_m
            writeOutput(initial_state, state, n, m, state.cost, r, sp)
            return 1
        
        neighbors = state.expand()

        for neighbor in reversed(neighbors):
            if (tuple(neighbor.config) not in frontier.configSet and tuple(neighbor.config) not in explored):
                frontier.push(neighbor)
                
        n += 1
        
    return 0


def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    frontier = Frontier(initial_state)
    explored = set()
    
    n=0 #nodes_expanded
    m = 0
    s  = time.time()
    s_m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    while (frontier.isEmpty() == 0):
        state = frontier.deleteMin()
        #print(state.config)
        if (state.cost > m):
            m = state.cost
            
        explored.add(tuple(state.config))
        
        if test_goal(state):
            r = time.time() - s
            sp = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - s_m
            writeOutput(initial_state, state, len(explored), m, state.cost, r, sp)
            return 1
        
        neighbors = state.expand()

        for neighbor in neighbors:
            if (tuple(neighbor.config) not in frontier.configSet and tuple(neighbor.config) not in explored):
                frontier.insert(neighbor)
            elif (tuple(neighbor.config) in frontier.configSet):
                frontier.decreaseKey(neighbor)
                
        n += 1
        
    return 0

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    total_man_dist = 0
    for idx, value in enumerate(state.config):
        if value != 0:
            total_man_dist += calculate_manhattan_dist(idx, value, state.n)
    
    return total_man_dist + state.cost

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    row = idx // n
    col = idx % n
    goal_row = value // n
    goal_col = value % n
    
    man_dist = abs(goal_row - row) + abs(goal_col - col)
    return man_dist
    
    pass

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    n = 0
    for i in puzzle_state.config:
        if i != n:
            return 0
        else:
            n += 1
        
    return 1

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()
