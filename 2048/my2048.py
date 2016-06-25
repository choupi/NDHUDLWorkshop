from term2048.game import Game
from term2048 import keypress
from term2048.board import Board
import random
import atexit
import time
import numpy
import math

class mGame(Game):
    def __init__(self, vis=False):
        if vis: Game.__init__(self, scores_file=None, store_file=None)
        self.best_score=0
        self.vis=vis
        self.reset()

    def reset(self):
        try: 
            print self.score, self.best_score, self.count, self.get_frame().max()
            del self.board
        except: pass
        #self.board = Board(**kws)
        self.board = Board(goal=512)
        self.score = 0
        self.count = 0
        self.moved = False
        self.pts = 0
        #self.clear_screen = clear_screen
        #self.__colors = colors
        #self.__azmode = azmode

    def play(self, action):
        self.moved = False
        pts=self.board.move(action+1)
        self.pts=pts
        self.incScore(pts)
        if pts>0: 
            self.count+=1
            self.moved=True
        if self.vis:
            margins = {'left': 4, 'top': 4, 'bottom': 4}
            self.clearScreen()
            print(self.__str__(margins=margins))
            time.sleep(0.1)
        
    def get_state(self):
        return self.board.cells

    def get_state_fake(self, action):
        b = Board()
        b.cells=self.board.cells
        b.move(action+1)
        return b.cells

    def get_score(self):
        if self.board.won(): return 1
        elif self.pts>0: return 1- 1.0/self.pts
        return 0
        #return self.score/2048.0
        #elif not self.board.canMove(): s=-1
        #elif not self.moved: s=-5
        #return self.count/80+math.log(float(self.get_frame().max()))/5.0+s
        #return self.count/100.0
        #return self.count/100.0+math.log(float(self.get_frame().max()))

    def is_over(self):
        return self.board.won() or not self.board.canMove()
    def is_won(self):
        return self.board.won()

    @property
    def name(self):
        return "2048"
    @property
    def nb_actions(self):
        return 4

    def get_frame(self):
        ll=numpy.vectorize(lambda x:math.log(x+1))
        #s=[self.get_state_fake(1),self.get_state_fake(2),self.get_state_fake(3),self.get_state_fake(4),self.get_state()]
        s=self.get_state()
        return ll(numpy.array(s).astype('float32'))

    def draw(self):
        return self.get_state()

class mrandomGame(Game):
    def __init__(self, step):
        Game.__init__(self, scores_file=None, store_file=None)
        self.mdirs=[1,2,3,4]
        self.step=step
        self.count=0
        self.cell=[]

    def loop(self):
        try:
            while True:
                if self.board.won() or not self.board.canMove():
                    break
                m = random.choice(self.mdirs)
                self.incScore(self.board.move(m))
                self.count+=1
                if self.count==self.step: self.cell=self.board.cells[:]
            if len(self.cell)>0: print '%d,%d,%s'%(self.score, self.count-self.step, ','.join([','.join(map(str, i)) for i in self.cell]))

        except KeyboardInterrupt:
            return

class randomGame(Game):
    def __init__(self):
        Game.__init__(self)
        self.mdirs=[1,2,3,4]

    def loop(self):
        pause_key = self.board.PAUSE
        margins = {'left': 4, 'top': 4, 'bottom': 4}
        atexit.register(self.showCursor)
        try:
            self.hideCursor()
            while True:
                self.clearScreen()
                #print self.board.cells
                print(self.__str__(margins=margins))
                if self.board.won() or not self.board.canMove():
                    break
                #m = self.readMove()
                m = random.choice(self.mdirs)
                self.incScore(self.board.move(m))

        except KeyboardInterrupt:
            #self.saveBestScore()
            return

if __name__ == '__main__':
    #for s in [70]:
    #    for i in xrange(50000):
    #        rg=mrandomGame(s)
    #        rg.loop()
    rg=randomGame()
    rg.loop()
