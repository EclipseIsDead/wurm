import pygame
import sys
import os
from pygame.locals import *

from worm import Worm

# python is such a worthless language
try:
    from brain import network
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from brain import network

if __name__ == '__main__':
    pygame.init()

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((800, 800))

    worm = Worm()
    brain = network.CElegansBrain()

    pygame.mouse.set_visible(0)
    pygame.display.set_caption('Wurm')

    while True:
        clock.tick(60)
        screen.fill([0, 0, 0])
        x, y = pygame.mouse.get_pos()

        motor_tensor = brain(5, 3)
        worm.move(motor_tensor)
        worm.draw(screen)

        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                pygame.display.update()
