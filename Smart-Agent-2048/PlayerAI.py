from random import randint
from BaseAI import BaseAI
from my_code import State
from my_code import Pruning
class BuildTree:
	# builds the game tree
    def __init__(self, gridCopy, steps=6):
        self.grid = gridCopy
        self.init_tree_value = '-1'
        self.tree_root = State(self.init_tree_value)
        self.steps = steps

    def build_it(self, i, root=None, grid=None):
        if (i == self.steps):
            return
        if (root is None):
            root = self.tree_root
        if (grid is None):
            grid = self.grid

        available_moves = grid.getAvailableMoves()
        root.init_successors()
        for move in available_moves:
            if grid.canMove([move]):
                tmp_grid = grid.clone()
                tmp_grid.move(move) 
                successor = State(value=str(tmp_grid.getMaxTile()), direction=move)
                root.add_successor(successor)

        my_stack = root.get_successors()
        my_stack_len = len(my_stack)
        my_stack = list(reversed(my_stack))
        for j in range(my_stack_len):
            successor = my_stack.pop()
            tmp_grid = grid.clone()
            tmp_grid.move(successor.get_direction())
            self.build_it(i+1, root=successor, grid=tmp_grid)

    def get_best_move(self):
        pruning_obj = Pruning(self.tree_root)
        return pruning_obj.init_pruning_tree()
class PlayerAI(BaseAI):
	# represents the AI player
	def getMove(self, grid):
		my_tree = BuildTree(grid)
		my_tree.build_it(0)
		return my_tree.get_best_move()