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
expl_anim = {}
expl_anim['lg'] = []
expl_anim['sm'] = []
expl_anim['player'] = []
for i in range(9):
    expl_img = load_image(os.path.join(BASE_PATH, "img", f"expl{i}.png"))
    # expl_img = pygame.image.load(os.path.join(BASE_PATH, "img", f"expl{i}.png"))
    expl_img.set_colorkey(BLACK)
    expl_anim['lg'].append(pygame.transform.scale(expl_img, (75, 75)))
    expl_anim['sm'].append(pygame.transform.scale(expl_img, (30, 30)))
    player_expl_img = load_image(os.path.join(BASE_PATH, "img", f"player_expl{i}.png"))
    # player_expl_img = pygame.image.load(os.path.join(BASE_PATH, "img", f"player_expl{i}.png"))
    player_expl_img.set_colorkey(BLACK)
    expl_anim['player'].append(player_expl_img)


class Explosion(pygame.sprite.Sprite):
    def __init__(self, center, size):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.image = expl_anim[self.size][0]
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.frame = 0
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = 50

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frame_rate:
            self.last_update = now
            self.frame += 1
            if self.frame == len(expl_anim[self.size]):
                self.kill()
            else:
                self.image = expl_anim[self.size][self.frame]
                center = self.rect.center
                self.rect = self.image.get_rect()
                self.rect.center = center