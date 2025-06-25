# Version 2 Game Environment

# %% [markdown]
# ## Importing the necessary libraries
import random
import pygame
import os

# %% [markdown]
# ## Setting
FPS = 60
WIDTH = 500 // 2
HEIGHT = 600 // 2

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Initialize the screen for convert_alpha()
pygame.display.set_mode((WIDTH, HEIGHT))

# Load image with convert_alpha()
def load_image(path, use_alpha=False):
    img = pygame.image.load(path)
    if pygame.display.get_surface():
        if use_alpha:
            return img.convert_alpha()
        else:
            return img.convert()
    return img 

BASE_PATH = os.path.join(os.getcwd(), 'space_ship_game_RL_V2')
# BASE_PATH = os.path.dirname(__file__)

# %% [markdown]
# ## Bullet Function

bullet_img = load_image(os.path.join(BASE_PATH, "img", "bullet.png"))
# bullet_img = pygame.image.load(os.path.join(BASE_PATH, "img", "bullet.png"))

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.rect = self.image.get_rect()
        self.image = pygame.transform.scale(self.image, (self.rect[2] // 1.2, self.rect[3] // 2))
        self.image.set_colorkey(BLACK)
        
        self.rect.centerx = x
        self.rect.bottom = y
        self.speedy = -10

    def update(self):
        self.rect.y += self.speedy
        if self.rect.bottom < 0:
            self.kill()

# %% [markdown]
# ## Explosion Function

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

# %% [markdown]
# ## Power Function

power_imgs = {}
power_imgs['shield'] = load_image(os.path.join(BASE_PATH, 'img', 'shield.png'))
power_imgs['shield'] = pygame.transform.scale(power_imgs['shield'], (20, 20))
# power_imgs['shield'] = pygame.image.load(os.path.join(BASE_PATH, 'img', 'shield.png'))
power_imgs['gun'] = load_image(os.path.join(BASE_PATH, 'img', 'gun.png'))
power_imgs['gun'] = pygame.transform.scale(power_imgs['gun'], (18, 25))
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

# %% [markdown]
# ## Rock Function

rock_imgs = []
for i in range(2, 7):
    img = load_image(os.path.join(BASE_PATH, "img", f"rock{i}.png"))
    w, h = img.get_rect().size
    img = pygame.transform.scale(img, (w//2, h//2))
    rock_imgs.append(img)

class Rock(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image_ori = random.choice(rock_imgs) 
        self.image_ori.set_colorkey(BLACK)
        self.image = self.image_ori.copy()
        self.rect = self.image.get_rect()
        self.radius = int(self.rect.width * 0.85 / 2)
        # pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(-180, -100)
        self.speedy = random.randrange(2, 4)
        self.speedx = random.randrange(-3, 3)
        self.total_degree = 0
        self.rot_degree = random.randrange(-3, 3)

    def rotate(self):
        self.total_degree += self.rot_degree
        self.total_degree = self.total_degree % 360
        self.image = pygame.transform.rotate(self.image_ori, self.total_degree)
        center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = center

    def update(self):
        # self.rotate()
        self.rect.y += self.speedy
        self.rect.x += self.speedx
        if self.rect.top > HEIGHT or self.rect.left > WIDTH or self.rect.right < 0:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(-100, -40)
            self.speedy = random.randrange(2, 10)
            self.speedx = random.randrange(-3, 3)

# %% [markdown]
# ## Player Function
# pygame.mixer.init()
player_img = load_image(os.path.join(BASE_PATH, "img", "player.png"))
# player_img = pygame.image.load(os.path.join(BASE_PATH, "img", "player.png"))
# shoot_sound = pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "shoot.wav"))

clock = pygame.time.Clock()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(player_img, (25, 19))
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
        self.is_shooting = False
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
            self.is_shooting = True
            self.bullet_ready = False
            self.shoot()
        else:
            self.is_shooting = False

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0

    def shoot(self):
        if not(self.hidden):
            if self.gun == 1:
                bullet = Bullet(self.rect.centerx, self.rect.top)
                self.bullet_group.add(bullet)
                # shoot_sound.play()
            elif self.gun >=2:
                bullet1 = Bullet(self.rect.left, self.rect.centery)
                bullet2 = Bullet(self.rect.right, self.rect.centery)
                self.bullet_group.add(bullet1)
                self.bullet_group.add(bullet2)
                # shoot_sound.play()

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

# %% [markdown]
# ## Game Class
# Initialize the game and create the window
pygame.init()
pygame.mixer.init()

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Load images
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
        self.last_collided_rock_radius = 0  # Add this attribute to store the radius of the last collided rock

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
            self.last_collided_rock_radius = hits[0].radius  # Store the radius of the collided rock
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

# %% [markdown]
