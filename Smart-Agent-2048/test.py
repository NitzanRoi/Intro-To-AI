# test - try to understand the game functions
from Grid       import Grid
from Displayer  import Displayer

from random import randint

class BuildTree:
    from my_code import State
    def __init__(self, gridCopy):
        self.grid = gridCopy
        self.available_moves = self.grid.getAvailableMoves() # for init state
        self.tree_root = State('0')
    def build_it(self): # todo from here: build it recursively - build level by level. dont forget to test all time
        last_root = self.tree_root
        for move in self.available_moves:
            if self.grid.canMove([move]):
                tmp_grid = self.grid.clone()
                tmp_grid.move(move)
                successor = State(tmp_grid.getMaxTile(), direction=move)
                last_root.add_successor(successor) # todo dont forget to add them sorted

    def get_best_move(self):
        self.build_it()

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
        # all the rest code is demo. here my code should be written
        
        print('AvailableCells:')
        print(self.grid.getAvailableCells()) # [(0, 1), (0, 2), (0, 3), (1, 0) ...]
        print('AvailableMoves:')
        print(self.grid.getAvailableMoves()) # [1, 2, 3] or [0, 1, 2, 3] ...
        
        # if self.grid.canMove([move]):
            # self.grid.move(move)
        # self.grid.getAvailableCells()
        # self.grid..getAvailableMoves()
        # self.grid.insertTile(pos, value) -> maybe redundant and 'move' is enough
        # self.grid.clone()
        # maxTile = self.grid.getMaxTile()
        # ACTIONS = {
        #     0: "UP",
        #     1: "DOWN",
        #     2: "LEFT",
        #     3: "RIGHT"
        # }
        
        return randint(0, 3)

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
    my_game = MyGame()
    my_game.start()

main()