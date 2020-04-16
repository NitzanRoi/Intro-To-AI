# test - use the game functions and understand them 
from Grid       import Grid
from Displayer  import Displayer
from random import randint

####
from my_code import State
from my_code import Pruning
class BuildTree:

    def __init__(self, gridCopy, steps=4):
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

        if (root.get_val() == self.init_tree_value):
            tmp_grid = grid
        else:
            tmp_grid = grid.clone()
            tmp_grid.move(root.get_direction())

        available_moves = tmp_grid.getAvailableMoves()
        root.init_successors()
        for move in available_moves:
            if tmp_grid.canMove([move]):
                sec_tmp_grid = grid.clone()
                sec_tmp_grid.move(move) 
                successor = State(value=str(sec_tmp_grid.getMaxTile()), direction=move)
                root.add_successor(successor)

        my_stack = root.get_successors()
        my_stack_len = len(my_stack)
        my_stack = list(reversed(my_stack))
        for j in range(my_stack_len):
            # print('i: ' + str(i))
            # print('root direction: ', str(root.get_direction()))
            successor = my_stack.pop()
            tmp_grid = grid.clone()
            tmp_grid.move(successor.get_direction())
            # print('successor direction: ', str(successor.get_direction()))
            self.build_it(i+1, root=successor, grid=tmp_grid)

    def get_best_move(self):
        pruning_obj = Pruning(self.tree_root)
        return pruning_obj.init_pruning_tree()

class MyGame:
    def __init__(self, size = 4):
        self.grid = Grid(size)
        self.possibleNewTiles = [2, 4]
        self.probability = 0.9
        self.initTiles = 2
        self.displayer = Displayer()
        self.timer = 0
        self.turns = 2

    def get_move(self, gridCopy):
        # all the rest code is demo. here is the important code:
        my_tree = BuildTree(gridCopy)
        my_tree.build_it(0)
        return my_tree.get_best_move() 
        
        # print('AvailableCells:')
        # print(self.grid.getAvailableCells()) # [(0, 1), (0, 2), (0, 3), (1, 0) ...]
        # print('AvailableMoves:')
        # print(self.grid.getAvailableMoves()) # [1, 2, 3] or [0, 1, 2, 3] ...
        
        # if self.grid.canMove([move]):
            # self.grid.move(move)
        # self.grid.getAvailableCells()
        # self.grid.getAvailableMoves()
        # self.grid.insertTile(pos, value) -> maybe redundant and 'move' is enough
        # self.grid.clone()
        # maxTile = self.grid.getMaxTile()
        # ACTIONS = {
        #     0: "UP",
        #     1: "DOWN",
        #     2: "LEFT",
        #     3: "RIGHT"
        # }
        
        # return randint(0, 3)

    def getNewTileValue(self):
        if randint(0,99) < 100 * self.probability:
            return self.possibleNewTiles[0]
        else:
            return self.possibleNewTiles[1];

    def insertRandonTile(self):
        tileValue = self.getNewTileValue()
        cells = self.grid.getAvailableCells()
        cell = cells[randint(0, len(cells) - 1)]
        self.grid.setCellValue(cell, tileValue)

    def start(self):
        for i in range(self.initTiles):
            self.insertRandonTile()
        self.displayer.display(self.grid)
        maxTile = 0
        while self.timer < self.turns:
            gridCopy = self.grid.clone()
            move = self.get_move(gridCopy)
            print('move: ', str(move))
            if self.grid.canMove([move]):
                self.grid.move(move)
                maxTile = self.grid.getMaxTile()
                self.timer += 1
            else:
                print("Invalid Move")
            self.displayer.display(self.grid)
        print('max tile:', str(maxTile))

def main():
    print('play')
    my_game = MyGame()
    my_game.start()

main()