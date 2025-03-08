import pygame
import os
import random
import numpy as np
import json
from neural_network import NeuralNetwork, GeneticAlgorithm  # Importando a rede neural e o algoritmo genético

pygame.init()

# Configurações gerais
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chrome Dino Runner Adaptado")
FONT_COLOR = (255, 255, 255)
FONT = pygame.font.Font("freesansbold.ttf", 20)

# Carregar imagens
RUNNING = [pygame.image.load(os.path.join("assets/Dino", f"DinoRun{i}.png")) for i in (1, 2)]
JUMPING = pygame.image.load(os.path.join("assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("assets/Dino", f"DinoDuck{i}.png")) for i in (1, 2)]
SMALL_CACTUS = [pygame.image.load(os.path.join("assets/Cactus", f"SmallCactus{i}.png")) for i in range(1, 4)]
LARGE_CACTUS = [pygame.image.load(os.path.join("assets/Cactus", f"LargeCactus{i}.png")) for i in range(1, 4)]
BIRD = [pygame.image.load(os.path.join("assets/Bird", f"Bird{i}.png")) for i in (1, 1)]
CLOUD = pygame.image.load(os.path.join("assets/Other", "Cloud.png"))
BG = pygame.image.load(os.path.join("assets/Other", "Track.png"))

# Classes
class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 10  # Aumentado para pulo mais rápido

    def __init__(self, neural_net):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING
        self.neural_net = neural_net  # Adiciona a rede neural ao dinossauro

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.on_ground = True  # Variável para controlar o status do salto
        
        self.alive = True

    def update(self, game_speed, obstacles):
        if len(obstacles) > 0:
            closest_obstacle = obstacles[0]
        else:
            closest_obstacle = Obstacle([SMALL_CACTUS[0]], 310, 15)  # Usar um obstáculo padrão

        y_obstacle = self.Y_POS - closest_obstacle.fake_y
        inputs = [game_speed, y_obstacle, closest_obstacle.rect.x, closest_obstacle.rect.width, closest_obstacle.rect.height]
        output = self.neural_net.forward(inputs)

        action = np.argmax(output)
        if action == 2 and self.on_ground:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
            self.on_ground = False
        elif action == 1:
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False
        elif action == 0 and self.on_ground:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False

        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 1.2

        if self.dino_rect.y >= self.Y_POS:
            self.dino_rect.y = self.Y_POS
            self.jump_vel = self.JUMP_VEL
            self.dino_jump = False
            self.on_ground = True

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Obstacle:
    def __init__(self, images, y_pos, y_pad=0):
        self.image = random.choice(images)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH
        self.rect.y = y_pos + y_pad
        self.fake_y = y_pos

    def update(self, game_speed, obstacles):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.remove(self)

    def draw(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y))

class Cloud:
    def __init__(self):
        self.image = CLOUD
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)

    def update(self, game_speed):
        self.x -= game_speed
        if self.x < -self.image.get_width():
            self.x = SCREEN_WIDTH + random.randint(800, 1000)
            self.y = random.randint(50, 100)

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        
class Button:
    def __init__(self, text, x, y, width, height, font, color, action=None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.font = font
        self.action = action

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surface = self.font.render(self.text, True, FONT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

def background(x_pos_bg, game_speed):
    image_width = BG.get_width()
    SCREEN.blit(BG, (x_pos_bg, 380))
    SCREEN.blit(BG, (image_width + x_pos_bg, 380))
    return -image_width if x_pos_bg <= -image_width else x_pos_bg - game_speed

def score_display(points):
    text = FONT.render(f"Points: {points}", True, FONT_COLOR)
    SCREEN.blit(text, (900, 50))

def epoch_display(epoch):
    text = FONT.render(f"Epoch: {epoch}", True, FONT_COLOR)
    SCREEN.blit(text, (50, 50))
    
def population_display(n_alive, n_population):
    text = FONT.render(f"Alive: {n_alive}/{n_population}", True, FONT_COLOR)
    SCREEN.blit(text, (50, 90))

def top_10_display(sorted_population):
    y_offset = 50
    for idx, network in enumerate(sorted_population[:10]):
        text = FONT.render(f"{network.id}: {network.current_score}", True, FONT_COLOR)
        SCREEN.blit(text, (900, y_offset))
        y_offset += 20
        

# Funções auxiliares
def save_population(epoch, population, filename="saved_population.json"):
    """Salva a população em um arquivo JSON, incluindo todos os pesos e biases."""
    data = {
        "epoch": epoch,
        "population": [
            {
                "id": p.id,
                "weights_input_hidden": p.weights_input_hidden.tolist(),
                "weights_hidden_output": p.weights_hidden_output.tolist(),
                "bias_hidden": p.bias_hidden.tolist(),
                "bias_output": p.bias_output.tolist(),
                "current_score": p.current_score,
                "best_score": p.best_score,
            } 
            for p in population
        ],
    }
    with open(filename, "w") as file:
        json.dump(data, file)

def load_population(filename="saved_population.json"):
    """Carrega a população de um arquivo JSON e recria os indivíduos."""
    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = json.load(file)
            population = []
            for individual in data["population"]:
                # Cria um novo objeto NeuralNetwork e carrega os valores salvos
                nn = NeuralNetwork(
                    input_size=len(individual["weights_input_hidden"]),
                    hidden_size=len(individual["bias_hidden"]),
                    output_size=len(individual["bias_output"])
                )
                nn.id = individual["id"]
                nn.weights_input_hidden = np.array(individual["weights_input_hidden"])
                nn.weights_hidden_output = np.array(individual["weights_hidden_output"])
                nn.bias_hidden = np.array(individual["bias_hidden"])
                nn.bias_output = np.array(individual["bias_output"])
                nn.current_score = individual["current_score"]
                nn.best_score = individual["best_score"]
                population.append(nn)
            return data["epoch"], population
    return 0, None

def start_screen():
    title_font = pygame.font.Font("freesansbold.ttf", 40)
    button_font = pygame.font.Font("freesansbold.ttf", 30)
    new_game_button = Button("New Game", SCREEN_WIDTH // 2 - 100, 250, 200, 50, button_font, (0, 128, 0))
    continue_button = Button("Continue", SCREEN_WIDTH // 2 - 100, 350, 200, 50, button_font, (128, 0, 0))

    while True:
        SCREEN.fill((0, 0, 0))
        title_surface = title_font.render("Dino AI Train", True, FONT_COLOR)
        SCREEN.blit(title_surface, (SCREEN_WIDTH // 2 - title_surface.get_width() // 2, 100))

        new_game_button.draw(SCREEN)
        continue_button.draw(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if new_game_button.is_clicked(event):
                return 0, None  # Começar novo jogo
            if continue_button.is_clicked(event):
                return load_population()  # Carregar população salva

        pygame.display.update()


# Função principal
def main():
    run = True
    clock = pygame.time.Clock()
    population_size = 100
    mutation_rate = 0.05
    generations = 1000
    
    ga = GeneticAlgorithm(population_size, mutation_rate)
    
    epoch_init = 0
    epoch_init, loaded_pop = start_screen()
    if loaded_pop != None:
        ga.population = loaded_pop
    
    for generation in range(epoch_init, generations):
        print(f"Geração: {generation + 1}")

        players = [Dinosaur(player) for player in ga.population]
        n_alives = population_size
        obstacles = []
        cloud = Cloud()
        x_pos_bg = 0
        game_speed = 20
        points = 0
        #fitness_scores = [0] * population_size
        

        while run:
            # Salvar a população
            
            SCREEN.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    save_population(generation, ga.population)

            for i, player in enumerate(players):
                if player.alive:
                    player.update(game_speed, obstacles)
                    player.draw(SCREEN)
                    ga.population[i].current_score += 1
                    #print(ga.population[i].current_score)
                #fitness_scores[i] += 1

                    for obstacle in list(obstacles):
                        if player.dino_rect.colliderect(obstacle.rect):
                            player.alive = False
                            n_alives -= 1
                            #players.remove(player)
                            break

            if len(obstacles) == 0:
                obstacle_type = random.choice([(SMALL_CACTUS, 310, 15), (LARGE_CACTUS, 310, -10), (BIRD, random.choice([310, 270, 210]))])
                obstacles.append(Obstacle(*obstacle_type))

            for obstacle in list(obstacles):
                obstacle.update(game_speed, obstacles)
                obstacle.draw(SCREEN)

            cloud.update(game_speed)
            cloud.draw(SCREEN)

            x_pos_bg = background(x_pos_bg, game_speed)

            #score_display(points)
            points += 1

            if points % 100 == 0:
                game_speed += 1
                
            sorted_population = sorted(ga.population, key=lambda x: x.current_score, reverse=True)
            top_10_display(sorted_population)
            #print([x.current_score for x in sorted_population[:10]])
            epoch_display(generation)
            population_display(n_alives, population_size)
            if n_alives == 0:
                for i, individual in enumerate(ga.population):
                    #individual.current_score = fitness_scores[i]
                    if individual.current_score > individual.best_score:
                        individual.best_score = individual.current_score

                break

            clock.tick(30)
            pygame.display.update()

        ga.evolve()

    pygame.quit()

main()



