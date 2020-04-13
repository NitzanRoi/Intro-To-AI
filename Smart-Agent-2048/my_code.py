## tests ##
import math

SUCCESS = 1
FAILURE = 0
ERROR = -1
CALC_CHILDREN_DONE = 9
EMPTY = 'Empty Value'

class Stack:
    # wrapping a stack data structure
    def __init__(self):
        self.stack = list()

    def push_stack(self, node):
        self.stack.append(node)

    def pop_stack(self):
        return self.stack.pop()

    def get_stack_size(self):
        return len(self.stack)

    def print_stack(self):
        print('The stack contains:')
        for i in self.stack:
            print(i.get_val())

class DFS:
    # calculating DFS on a given tree
    def __init__(self, goal_state, init_state):
        self.stack = Stack()
        self.init_state_children = list()
        self.goal_state = goal_state
        self.init_state = init_state

    def print_or_children(self, is_search, cur_state):
        if (is_search):
            print(cur_state.get_val())
        else:
           self.init_state_children.append(cur_state) 

    def calc_dfs(self, is_search, cur_state=False):
        if (self.stack.get_stack_size() == 1 and
            not cur_state.get_right_successor() and
            not cur_state.get_left_successor() and
            not cur_state.get_val() == self.goal_state.get_val()): # (has no successors and isn't the goal)
            self.print_or_children(is_search, cur_state)
            return FAILURE if is_search else CALC_CHILDREN_DONE
        if (cur_state == False):
            cur_state = self.init_state
            self.stack.push_stack(cur_state)
        if (not self.goal_state or cur_state is None):
            print('error')
            return ERROR
        if (cur_state.get_val() == self.goal_state.get_val()):
            self.print_or_children(is_search, cur_state)
            return SUCCESS
        self.print_or_children(is_search, cur_state)
        if (cur_state.get_right_successor()):
            self.stack.push_stack(cur_state.get_right_successor())
        if (cur_state.get_left_successor()):
            self.stack.push_stack(cur_state.get_left_successor())
        return self.calc_dfs(is_search, self.stack.pop_stack())

    # def get_all_children(self, init_state=None):
    #     if (init_state is None):
    #         print('please fill init state')
    #         return FAILURE
    #     else:
    #         self.init_state = init_state # node to start DFS with
    #         self.goal_state = State(EMPTY)
    #     dfs_result = self.calc_dfs(is_search=False)
    #     self.stack = Stack()
    #     if (dfs_result == CALC_CHILDREN_DONE):
    #         return self.init_state_children
    #     return dfs_result

class State:
    # node in the tree
    # todo: currently binary tree but maybe should be extended 
    # to have more than 2 children in each level? 
    # possible, but first implement the alpha-beta. If the DFS isn't neccesary, so dont change it.
    # Change as few as possible (maybe change only the Minimax)..
    def __init__(self, value, left_successor=None, right_successor=None):
        self.value = value
        self.left_successor = left_successor
        self.right_successor = right_successor

    def get_val(self):
        return int(self.value)

    def get_left_successor(self):
        return self.left_successor

    def get_right_successor(self):
        return self.right_successor

class MiniMax:
    # MiniMax algorithm
    def __init__(self):
        self.init_tree()

    def init_tree(self): 
        # todo fill it correctly. 
        # first DEBUG
        # then understand how to use this function to the 2048 game
        
        o = State('5')
        n = State('3')
        m = State('9')
        l = State('8')
        k = State('4')
        j = State('7')
        i = State('9')
        h = State('2')
        g = State('16', n, o)
        f = State('15', l, m)
        e = State('14', j, k)
        d = State('13', h , i)
        c = State('12', f, g)
        b = State('11', d, e)
        a = State('10', b, c)
        decision = self.decision(a)
        print('decision (max child): ' + str(decision.get_val()))

    def get_node_children(self, node):
        return [node.get_left_successor(), node.get_right_successor()]

    def is_state_leaf(self, cur_state):
        return (cur_state.get_left_successor() is None and 
                cur_state.get_right_successor() is None)

    def decision(self, cur_state):
        max_child, max_utility = self.maximize(cur_state)
        print("max utility: " + str(max_utility))
        return max_child

    def maximize(self, cur_state):
        if (self.is_state_leaf(cur_state)):
            return None, cur_state.get_val()
        max_child, max_utility = None, -math.inf
        for child in self.get_node_children(cur_state):
            if child is None:
                continue
            min_child, min_utility = self.minimize(child)
            if (min_utility > max_utility):
                max_child, max_utility = child, min_utility
        return max_child, max_utility

    def minimize(self, cur_state):
        if (self.is_state_leaf(cur_state)):
            return None, cur_state.get_val()
        min_child, min_utility = None, math.inf
        for child in self.get_node_children(cur_state):
            if child is None:
                continue
            max_child, max_utility = self.maximize(child)
            if (max_utility < min_utility):
                min_child, min_utility = child, max_utility
        return min_child, min_utility

class Pruning: # TODO from here
    # Alpha - Beta pruning algorithm
    def __init__(self):
        pass

def main():
    mm = MiniMax()
    
    # currently problem with max recursion. debug it in a notebook
    # x = State('33333')
    # g = State('6')
    # f = State('5')
    # e = State('4')
    # d = State('3')
    # c = State('2', f, g)
    # b = State('1',d ,e)
    # a = State('0', b, c)
    
    # dfs = DFS(c, a) # params: goal_state, init_state

    # choose option: get children or search node (goal_state):

    # print('all children - dfs order')
    # res = dfs.get_children(init_state=a)
    # for i in res:
    #     print(i.get_val())

    # print('search')
    # res = dfs.calc_dfs(is_search=True)
    # print('res: ' + str(res))
 
main()
