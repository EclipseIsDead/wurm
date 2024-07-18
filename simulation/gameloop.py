import pygame
import sys
import os

from pygame.locals import *

pygame.init()

clock = pygame.time.Clock()
screen = pygame.display.set_mode((800, 800))

pygame.mouse.set_visible(0)
pygame.display.set_caption('Wurm')

while True:
    clock.tick(60)
    screen.fill([0, 0, 0])
    x, y = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
            pygame.display.update()
