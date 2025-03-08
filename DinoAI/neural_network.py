import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.randn(self.hidden_size)
        self.bias_output = np.random.randn(self.output_size)
        
        self.id = None
        self.current_score = 0
        self.best_score = 0

    def forward(self, inputs):
        inputs = self.min_max_scaling(inputs)
        self.hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden = self.relu(self.hidden)

        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        self.output = self.softmax(self.output)
        return self.output

    def min_max_scaling(self, inputs):
        min_val = np.min(inputs)
        max_val = np.max(inputs)
        scaled_inputs = (inputs - min_val) / (max_val - min_val)
        return scaled_inputs

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [NeuralNetwork(5, 10, 3) for _ in range(self.population_size)]
        
        self.id_counter = 0  # Contador global de IDs
        
        for i, network in enumerate(self.population):
            network.id = f'Player {self.id_counter}'
            self.id_counter += 1

    def alternative_evolve(self):
            sorted_population = sorted(self.population, key=lambda x: x.current_score, reverse=True)
            
            # Manter os 10% melhores diretamente
            top_10_percent = sorted_population[:self.population_size // 10]
            
            # Fazer crossover entre os 30% melhores
            top_30_percent = sorted_population[self.population_size // 10:self.population_size * 4 // 10]
            next_30_percent = sorted_population[self.population_size * 4 // 10:self.population_size * 7 // 10]
            
            new_population = top_10_percent.copy()
            
            for _ in range((self.population_size * 3 // 10) // 2):
                parent1, parent2 = random.sample(top_30_percent, 2)
                new_population.append(self.crossover(parent1, parent2))
            
            for _ in range((self.population_size * 3 // 10) // 2):
                parent1, parent2 = random.sample(next_30_percent, 2)
                new_population.append(self.crossover(parent1, parent2))
            
            # Selecionar o restante aleatoriamente
            remaining = sorted_population[self.population_size * 7 // 10:]
            while len(new_population) < self.population_size:
                new_population.append(random.choice(remaining))
            
            self.population = new_population
            
            for network in self.population:
                network.current_score = 0

    def evolve(self):
        sorted_population = sorted(self.population, key=lambda x: x.current_score, reverse=True)
        
        # Manter os 10% melhores diretamente
        top_10_percent = sorted_population[:self.population_size // 10]
        
        new_population = top_10_percent.copy()
        
        # Fazer os dois melhores fazerem crossover com 30% da população cada
        top_2 = sorted_population[:2]
        next_30_percent = sorted_population[2:self.population_size * 3 // 10]
        
        for _ in range((self.population_size * 3 // 10)):
            parent1 = top_2[0]
            parent2 = random.choice(next_30_percent)
            new_population.append(self.mutate(self.crossover(parent1, parent2)))
        
        for _ in range((self.population_size * 3 // 10)):
            parent1 = top_2[1]
            parent2 = random.choice(next_30_percent)
            new_population.append(self.mutate(self.crossover(parent1, parent2)))
        
        # Fazer crossovers aleatoriamente para o restante
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.population, 2)
            new_population.append(self.mutate(self.crossover(parent1, parent2)))
        
        self.population = new_population
        
        for network in self.population:
            network.current_score = 0

    def alternative_evolve2(self):
        sorted_population = sorted(self.population, key=lambda x: x.current_score, reverse=True)
        self.population = sorted_population[:self.population_size // 2]
        while len(self.population) < self.population_size:
            parent1, parent2 = random.sample(self.population[:self.population_size // 2], 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            self.population.append(child)

        for network in self.population:
            network.current_score = 0


    def crossover_ver2(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
        child.weights_input_hidden = (parent1.weights_input_hidden + parent2.weights_input_hidden) / 2
        child.weights_hidden_output = (parent1.weights_hidden_output + parent2.weights_hidden_output) / 2
        child.bias_hidden = (parent1.bias_hidden + parent2.bias_hidden) / 2
        child.bias_output = (parent1.bias_output + parent2.bias_output) / 2
        child.id = parent1.id
        return child
    
    def crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
        
        # Crossover para weights_input_hidden
        child.weights_input_hidden = np.where(np.random.rand(*parent1.weights_input_hidden.shape) < 0.5,
                                              parent1.weights_input_hidden,
                                              parent2.weights_input_hidden)
        
        # Crossover para weights_hidden_output
        child.weights_hidden_output = np.where(np.random.rand(*parent1.weights_hidden_output.shape) < 0.5,
                                               parent1.weights_hidden_output,
                                               parent2.weights_hidden_output)
        
        # Crossover para bias_hidden
        child.bias_hidden = np.where(np.random.rand(*parent1.bias_hidden.shape) < 0.5,
                                     parent1.bias_hidden,
                                     parent2.bias_hidden)
        
        # Crossover para bias_output
        child.bias_output = np.where(np.random.rand(*parent1.bias_output.shape) < 0.5,
                                     parent1.bias_output,
                                     parent2.bias_output)
        
        # Atribuir ID único ao filho
        child.id = f'Player {self.id_counter}'
        self.id_counter += 1
        return child

    def mutate_ver2(self, neural_net):
        if random.random() < self.mutation_rate:
            mutation_mask = np.random.rand(*neural_net.weights_input_hidden.shape) < self.mutation_rate
            neural_net.weights_input_hidden += np.random.randn(*neural_net.weights_input_hidden.shape) * mutation_mask

            mutation_mask = np.random.rand(*neural_net.weights_hidden_output.shape) < self.mutation_rate
            neural_net.weights_hidden_output += np.random.randn(*neural_net.weights_hidden_output.shape) * mutation_mask

            mutation_mask = np.random.rand(*neural_net.bias_hidden.shape) < self.mutation_rate
            neural_net.bias_hidden += np.random.randn(*neural_net.bias_hidden.shape) * mutation_mask

            mutation_mask = np.random.rand(*neural_net.bias_output.shape) < self.mutation_rate
            neural_net.bias_output += np.random.randn(*neural_net.bias_output.shape) * mutation_mask
                   
    def mutate(self, neural_net):
        for i in range(neural_net.weights_input_hidden.shape[0]):
            for j in range(neural_net.weights_input_hidden.shape[1]):
                if random.random() < self.mutation_rate:
                    mutation_type = random.choice(['add', 'mul', 'replace'])
                    if mutation_type == 'add':
                        neural_net.weights_input_hidden[i, j] += np.random.randn() * 0.1
                    elif mutation_type == 'mul':
                        neural_net.weights_input_hidden[i, j] *= 1 + (np.random.randn() * 0.1)
                    elif mutation_type == 'replace':
                        neural_net.weights_input_hidden[i, j] = np.random.randn()

        for i in range(neural_net.weights_hidden_output.shape[0]):
            for j in range(neural_net.weights_hidden_output.shape[1]):
                if random.random() < self.mutation_rate:
                    mutation_type = random.choice(['add', 'mul', 'replace'])
                    if mutation_type == 'add':
                        neural_net.weights_hidden_output[i, j] += np.random.randn() * 0.1
                    elif mutation_type == 'mul':
                        neural_net.weights_hidden_output[i, j] *= 1 + (np.random.randn() * 0.1)
                    elif mutation_type == 'replace':
                        neural_net.weights_hidden_output[i, j] = np.random.randn()

        for i in range(neural_net.bias_hidden.shape[0]):
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['add', 'mul', 'replace'])
                if mutation_type == 'add':
                    neural_net.bias_hidden[i] += np.random.randn() * 0.1
                elif mutation_type == 'mul':
                    neural_net.bias_hidden[i] *= 1 + (np.random.randn() * 0.1)
                elif mutation_type == 'replace':
                    neural_net.bias_hidden[i] = np.random.randn()

        for i in range(neural_net.bias_output.shape[0]):
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['add', 'mul', 'replace'])
                if mutation_type == 'add':
                    neural_net.bias_output[i] += np.random.randn() * 0.1
                elif mutation_type == 'mul':
                    neural_net.bias_output[i] *= 1 + (np.random.randn() * 0.1)
                elif mutation_type == 'replace':
                    neural_net.bias_output[i] = np.random.randn()
        
        return neural_net