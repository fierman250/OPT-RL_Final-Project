import pygame
import random
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
power_imgs = {}
power_imgs['shield'] = load_image(os.path.join(BASE_PATH, 'img', 'shield.png'))
# power_imgs['shield'] = pygame.image.load(os.path.join(BASE_PATH, 'img', 'shield.png'))
power_imgs['gun'] = load_image(os.path.join(BASE_PATH, 'img', 'gun.png'))
# power_imgs['gun'] = pygame.image.load(os.path.join(BASE_PATH, 'img', 'gun.png'))


class Power(pygame.sprite.Sprite):
    def __init__(self, center):
        pygame.sprite.Sprite.__init__(self)
        self.type = random.choice(['shield', 'gun'])
        self.image = power_imgs[self.type]
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.speedy = 3

    def update(self):
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT:
            self.kill()