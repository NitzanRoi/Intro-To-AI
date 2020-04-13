## my code ##
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
    # calculating DFS on a given tree (currently only binary tree)
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

    def push_stack_list_of_nodes(self, cur_state):
        if (cur_state.has_successors()):
            for successor in cur_state.get_successors():
                self.stack.push_stack(successor)

    def calc_dfs(self, is_search, cur_state=False):
        # if current state has no successors and isn't the goal:
        if (self.stack.get_stack_size() == 1 and
            not cur_state.has_successors() and
            not cur_state.get_val() == self.goal_state.get_val()):
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
        self.push_stack_list_of_nodes(cur_state)
        return self.calc_dfs(is_search, self.stack.pop_stack())

    def get_all_children(self, init_state=None):
        if (init_state is None):
            print('please fill init state')
            return FAILURE
        else:
            self.init_state = init_state # node to start the DFS with
            self.goal_state = State(EMPTY)
        dfs_result = self.calc_dfs(is_search=False)
        self.stack = Stack()
        if (dfs_result == CALC_CHILDREN_DONE):
            return self.init_state_children
        return dfs_result

class State:
    # represents a node in the tree

    # initialize list of children sorted 
    # from the most right successor (should be index 0)
    # to the most left successor (should be last index) 
    def __init__(self, value, successors=[]):
        self.value = value
        self.successors = successors

    def get_val(self):
        return self.value

    def has_successors(self):
        return len(self.successors) > 0

    def get_successors(self):
        return self.successors

class MiniMax:
    # MiniMax algorithm
    def __init__(self):
        self.init_minimax_tree()

    def init_minimax_tree(self): 
        # todo fill it correctly
        # understand how to use this function to the 2048 game
        
        o = State('5')
        n = State('3')
        m = State('9')
        l = State('10')
        k = State('5')
        j = State('2')
        i = State('1')
        h = State('7')
        g = State('9')
        f = State('15')
        e = State('3')
        d = State('3', [l, k, j, i])
        c = State('2', [h, g])
        b = State('1', [e])
        a = State('0', [d, c, b])
        decision = self.decision(a)
        print('decision (max child): ' + str(decision.get_val()))

    def get_node_children(self, node):
        return list(reversed(node.get_successors()))

    def is_state_leaf(self, cur_state):
        return not cur_state.has_successors()

    def decision(self, cur_state):
        max_child, max_utility = self.maximize(cur_state)
        print("max utility: " + str(max_utility))
        return max_child

    def maximize(self, cur_state):
        if (self.is_state_leaf(cur_state)):
            return None, int(cur_state.get_val())
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
            return None, int(cur_state.get_val())
        min_child, min_utility = None, math.inf
        for child in self.get_node_children(cur_state):
            if child is None:
                continue
            max_child, max_utility = self.maximize(child)
            if (max_utility < min_utility):
                min_child, min_utility = child, max_utility
        return min_child, min_utility

class Pruning:
    # Alpha - Beta pruning algorithm
    def __init__(self):
        self.init_pruning_tree()

    def init_pruning_tree(self):
        # todo fill it correctly
        # understand how to use this function to the 2048 game

        # o = State('5')
        # n = State('3')
        m = State('2')
        l = State('5')
        k = State('14')
        j = State('0')
        i = State('7')
        h = State('2')
        g = State('8')
        f = State('12')
        e = State('3')
        d = State('21', [m, l, k])
        c = State('11', [j, i, h])
        b = State('1', [g, f, e])
        a = State('6', [d, c, b])

        decision = self.decision(a)
        print('decision (max child): ' + str(decision.get_val()))

    def get_node_children(self, node):
        return list(reversed(node.get_successors()))

    def is_state_leaf(self, cur_state):
        return not cur_state.has_successors()

    def decision(self, cur_state):
        alpha = -math.inf
        beta = math.inf
        max_child, max_utility = self.maximize(cur_state, alpha, beta)
        print("max utility: " + str(max_utility))
        return max_child

    def maximize(self, cur_state, alpha, beta):
        if (self.is_state_leaf(cur_state)):
            return None, int(cur_state.get_val())
        max_child, max_utility = None, -math.inf
        for child in self.get_node_children(cur_state):
            if child is None:
                continue
            min_child, min_utility = self.minimize(child, alpha, beta)
            if (min_utility > max_utility):
                max_child, max_utility = child, min_utility
            if (max_utility >= beta):
                break
            if (max_utility > alpha):
                alpha = max_utility
        return max_child, max_utility

    def minimize(self, cur_state, alpha, beta):
        if (self.is_state_leaf(cur_state)):
            return None, int(cur_state.get_val())
        min_child, min_utility = None, math.inf
        for child in self.get_node_children(cur_state):
            if child is None:
                continue
            max_child, max_utility = self.maximize(child, alpha, beta)
            if (max_utility < min_utility):
                min_child, min_utility = child, max_utility
            if (min_utility <= alpha):
                break
            if (min_utility < beta):
                beta = min_utility
        return min_child, min_utility

class Test:
    # class for tests
    def __init__(self):
        pass

    def init_dfs_test_tree(self):
        o = State('5')
        n = State('3')
        m = State('9')
        l = State('8')
        k = State('4')
        j = State('79')
        i = State('5')
        h = State('4')
        g = State('8')
        f = State('7')
        e = State('6')
        d = State('3', [i, h])
        c = State('2', [g, f, e])
        b = State('1', [d])
        a = State('0', [c, b])
        self.goal_state = j
        self.init_state = a
        self.dfs = DFS(self.goal_state, self.init_state)

    def test_dfs_print_children(self):
        print('print all children - dfs order')
        res = self.dfs.get_all_children(init_state=self.init_state)
        for i in res:
            print(i.get_val())

    def test_dfs_search_and_print(self):
        print('search the goal state with value: ', self.goal_state.get_val())
        res = self.dfs.calc_dfs(is_search=True)
        print('res: ' + str(res))

    def test_minimax(self):
        print('--- Test MiniMax ---')
        mm = MiniMax()
        print('--- End test MiniMax ---')

    def test_pruning(self):
        print('--- Test Pruning ---')
        prun = Pruning()
        print('--- End test Pruning ---')

def main():
    test_obj = Test()

    # dont run print and search together:
    # test_obj.init_dfs_test_tree()
    # test_obj.test_dfs_print_children()
    # test_obj.test_dfs_search_and_print()

    # test_obj.test_minimax()
    # test_obj.test_pruning()

if __name__ == "__main__":
    main()
