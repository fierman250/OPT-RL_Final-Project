# Space Survival Game
import pygame
import random 
import os
from setting import *

# Game initialization and window creation
# pygame.init()
# pygame.mixer.init()
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_mode((WIDTH, HEIGHT))

from power import Power
from explosion import Explosion
from rock import Rock
from player import Player


# Load images
BASE_PATH = os.path.dirname(__file__)

background_img = pygame.image.load(os.path.join(BASE_PATH, "img", "background.png"))

font_name = os.path.join(BASE_PATH, "font.ttf")

class Game:
    def __init__(self):
        self.running = True
        self.player = pygame.sprite.GroupSingle()
        self.player.add(Player())
        self.all_sprites = pygame.sprite.Group()
        self.rocks = pygame.sprite.Group()
        for i in range(8):
            self.new_rock()
        self.powers = pygame.sprite.Group()

        self.score = 0
        self.surface = pygame.Surface((WIDTH, HEIGHT))  # Used for off-screen drawing
        self.state = pygame.surfarray.array3d(self.surface) # shape (500, 600, 3)
        self.action = 0

        self.is_collided = False
        self.is_hit_rock = False
        self.is_power = False

    def draw_text(self, surf, text, size, x, y):
        font = pygame.font.Font(font_name, size)
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.centerx = x
        text_rect.top = y
        surf.blit(text_surface, text_rect)

    def draw_health(self, surf, hp, x, y):
        if hp < 0:
            hp = 0
        BAR_LENGTH = 100
        BAR_HEIGHT = 10
        fill = (hp/100)*BAR_LENGTH
        outline_rect = pygame.Rect(x, y, BAR_LENGTH, BAR_HEIGHT)
        fill_rect = pygame.Rect(x, y, fill, BAR_HEIGHT)
        pygame.draw.rect(surf, GREEN, fill_rect)
        pygame.draw.rect(surf, WHITE, outline_rect, 2)

    def new_rock(self):
        r = Rock()
        self.all_sprites.add(r)
        self.rocks.add(r)

    def collide_bullet_rock(self):
        # Check bullet-rock collision
        hits = pygame.sprite.groupcollide(self.rocks, self.player.sprite.bullet_group, True, True)
        if hits:
            for hit in hits:
                # random.choice(expl_sounds).play()
                self.score += hit.radius * 2
                expl = Explosion(hit.rect.center, 'lg')
                self.all_sprites.add(expl)
                if random.random() > 0.8:
                    pow = Power(hit.rect.center)
                    self.all_sprites.add(pow)
                    self.powers.add(pow)
                self.new_rock()
            
            self.is_hit_rock = True

        else:
            self.is_hit_rock = False

    def collide_player_rock(self):
        # Check spaceship-rock collision
        hits = pygame.sprite.spritecollide(self.player.sprite, self.rocks, True, pygame.sprite.collide_circle)
        if hits:
            self.player.sprite.health -= hits[0].radius * 1
            expl = Explosion(hits[0].rect.center, 'sm')
            self.all_sprites.add(expl)
            self.new_rock()
            self.is_collided = True
        else:
            self.is_collided = False

    def collide_player_power(self):
        # Check spaceship-powerup collision
        hits = pygame.sprite.spritecollide(self.player.sprite, self.powers, True)
        if hits:
            for hit in hits:
                if hit.type == 'shield':
                    self.player.sprite.health += 20
                    if self.player.sprite.health > 100:
                        self.player.sprite.health = 100
                elif hit.type == 'gun':
                    self.player.sprite.gunup()
            self.is_power = True
        else:
            self.is_power = False

    def check_state(self):
        if self.player.sprite.health <= 0:
            death_expl = Explosion(self.player.sprite.rect.center, 'player')
            self.all_sprites.add(death_expl)
            
            self.player.sprite.lives -= 1
            # self.player.sprite.health = 100
            self.player.sprite.hide()

        if self.player.sprite.lives == 0:
            # die_sound.play()
            self.running = False

    def update(self, action):
        # Update game
        self.all_sprites.update()
        self.player.update(action)
        self.collide_bullet_rock()
        self.collide_player_rock()
        self.collide_player_power()
        self.check_state()

    def draw(self, screen=None):
        surface = self.surface if screen is None else screen
        surface.fill(BLACK)
        surface.blit(background_img, (0, 0))
        self.all_sprites.draw(surface)
        self.player.draw(surface)
        self.player.sprite.bullet_group.draw(surface)
        self.draw_text(surface, str(self.score), 18, WIDTH/2, 10)
        self.draw_health(surface, self.player.sprite.health, 5, 15)

        # Update state
        self.state = pygame.surfarray.array3d(surface)

        # Only update window in render mode
        if screen is not None:
            pygame.display.update()
