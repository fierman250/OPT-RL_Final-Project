import pygame
import random
import os
from setting import * 
from bullet import Bullet

def load_image(path, use_alpha=False):
    img = pygame.image.load(path)
    if pygame.display.get_surface():
        if use_alpha:
            return img.convert_alpha()
        else:
            return img.convert()
    return img 

BASE_PATH = os.path.dirname(__file__)
player_img = load_image(os.path.join(BASE_PATH, "img", "player.png"))
# player_img = pygame.image.load(os.path.join(BASE_PATH, "img", "player.png"))

shoot_sound = pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "shoot.wav"))

clock = pygame.time.Clock()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(player_img, (50, 38))
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.radius = 20
        # pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 10
        self.speedx = 8
        self.health = 100
        self.lives = 1
        self.hidden = False
        self.hide_time = 0
        self.gun = 1
        self.gun_time = 0
        self.bullet_group = pygame.sprite.Group()
        self.bullet_ready = True
        # self.bullet_time = 0
        self.bullet_timer = []
        self.dt = clock.tick(FPS)  # 回傳毫秒
        self.bullet_delay = 60 # 每60偵射一次 one shoot / 60 frames

    def update(self, action):
        self.bullet_group.update()
        self.recharge_bullet()

        now = pygame.time.get_ticks()
        if self.gun > 1 and now - self.gun_time > 5000:
            self.gun -= 1
            self.gun_time = now

        if self.hidden and now - self.hide_time > 1000:
            self.hidden = False
            self.rect.centerx = WIDTH / 2
            self.rect.bottom = HEIGHT - 10

        if action == 0:
            pass

        if action == 1:
            self.rect.x -= self.speedx
        if action == 2:
            self.rect.x += self.speedx

        if action == 3 and self.bullet_ready:
            self.bullet_ready = False
            self.shoot()
            # self.bullet_time = pygame.time.get_ticks()

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0

    def shoot(self):
        if not(self.hidden):
            if self.gun == 1:
                bullet = Bullet(self.rect.centerx, self.rect.top)
                self.bullet_group.add(bullet)
                shoot_sound.play()
            elif self.gun >=2:
                bullet1 = Bullet(self.rect.left, self.rect.centery)
                bullet2 = Bullet(self.rect.right, self.rect.centery)
                self.bullet_group.add(bullet1)
                self.bullet_group.add(bullet2)
                shoot_sound.play()

    def recharge_bullet(self):    
        self.bullet_timer.append(self.dt)
        if len(self.bullet_timer) == self.bullet_delay:
            self.bullet_ready = True
            self.bullet_timer = []   

    def hide(self):
        self.hidden = True
        self.hide_time = pygame.time.get_ticks()
        self.rect.center = (WIDTH/2, HEIGHT+500)

    def gunup(self):
        self.gun += 1
        self.gun_time = pygame.time.get_ticks()