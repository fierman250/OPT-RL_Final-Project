import pygame
import os
from setting import * 

def load_image(path, use_alpha=False):
    img = pygame.image.load(path)
    if pygame.display.get_surface():
        if use_alpha:
            return img.convert_alpha()
        else:
            return img.convert()
    return img 

BASE_PATH = os.path.dirname(__file__)

bullet_img = load_image(os.path.join(BASE_PATH, "img", "bullet.png"))
# bullet_img = pygame.image.load(os.path.join(BASE_PATH, "img", "bullet.png"))

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.bottom = y
        self.speedy = -10

    def update(self):
        self.rect.y += self.speedy
        if self.rect.bottom < 0:
            self.kill()