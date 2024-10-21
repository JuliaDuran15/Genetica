import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import time

# Definindo os parâmetros do problema
num_points = 10  # número de pontos (pelo menos 8)
pop_size = 100  # tamanho da população (quantidade de "soluções" simultâneas)
max_generations = 1000  # número máximo de gerações (ou ciclos de melhoria)
initial_mutation_rate = 0.1  # taxa de mutação inicial (10%)
patience = 200  # número de gerações sem melhoria antes de parar

# Gerando pontos aleatórios para os dois cenários
np.random.seed(42)  # para garantir que os resultados sejam reproduzíveis
points_uniform = np.random.rand(num_points, 2) * 100  # pontos distribuídos uniformemente
theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
points_circle = np.c_[50 + 40 * np.cos(theta), 50 + 40 * np.sin(theta)]  # pontos dispostos em círculo

# Função auxiliar para calcular a distância total de um caminho
def calculate_distance(path, points):
    return sum(euclidean(points[path[i]], points[path[i + 1]]) for i in range(len(path) - 1)) + \
           euclidean(points[path[-1]], points[path[0]])

# Função de aptidão (quanto melhor a solução, maior a aptidão)
def fitness(path, points):
    return 1 / calculate_distance(path, points)  # quanto menor a distância, maior a aptidão

# Criação da população inicial (sequências aleatórias de pontos)
def create_initial_population(pop_size, num_points):
    return [random.sample(range(num_points), num_points) for _ in range(pop_size)]

# Seleção (usando o método do torneio)
def tournament_selection(population, points, k=5):
    selected = random.sample(population, k)  # escolhe 5 indivíduos aleatoriamente
    selected.sort(key=lambda ind: fitness(ind, points), reverse=True)  # ordena do melhor para o pior
    return selected[0]  # retorna o melhor indivíduo

# Cruzamento do tipo Order Crossover (OX)
def order_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))  # escolhe dois pontos de corte aleatórios
    child = [-1] * size
    child[a:b] = parent1[a:b]  # copia uma parte do primeiro pai para o filho

    fill_pos = b
    for elem in parent2:
        if elem not in child:  # preenche o restante do filho com genes do segundo pai
            if fill_pos >= size:
                fill_pos = 0
            child[fill_pos] = elem
            fill_pos += 1
    return child

# Mutação (troca duas posições aleatoriamente)
def mutate(path, mutation_rate):
    if random.random() < mutation_rate:  # aplica mutação com uma certa chance
        i, j = random.sample(range(len(path)), 2)  # escolhe duas posições aleatórias
        path[i], path[j] = path[j], path[i]  # troca os elementos de posição
    return path

# Função para alterar a taxa de mutação dinamicamente a cada 1/4 das gerações
def adjust_mutation_rate(generation, max_generations, initial_rate):
    quarter = max_generations // 4
    if generation < quarter:
        return initial_rate  # Primeiro quarto de gerações
    elif generation < 2 * quarter:
        return initial_rate * 1.5  # Aumenta a taxa de mutação no segundo quarto
    elif generation < 3 * quarter:
        return initial_rate * 0.5  # Reduz a taxa de mutação no terceiro quarto
    else:
        return initial_rate * 0.25  # Reduz ainda mais a taxa de mutação no último quarto

# Executando o Algoritmo Genético com critério de parada
def genetic_algorithm(points, pop_size, max_generations, initial_mutation_rate, patience, scenario_name=""):
    # Criação da população inicial
    population = create_initial_population(pop_size, len(points))
    best_solution = None
    best_distance = float('inf')  # inicializa com uma distância muito grande
    history = []  # para armazenar a evolução da solução ao longo das gerações
    generations_without_improvement = 0  # contador de gerações sem melhoria

    # Iterando ao longo das gerações
    for generation in range(max_generations):
        # Ajusta a taxa de mutação dinamicamente
        mutation_rate = adjust_mutation_rate(generation, max_generations, initial_mutation_rate)
        
        new_population = []  # nova geração
        for _ in range(pop_size // 2):
            # Seleciona os pais e faz cruzamento
            parent1 = tournament_selection(population, points)
            parent2 = tournament_selection(population, points)
            child1 = order_crossover(parent1, parent2)
            child2 = order_crossover(parent2, parent1)
            # Adiciona os filhos mutados à nova população
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        
        # Substitui a população antiga pela nova
        population = new_population
        # Encontra o melhor indivíduo da geração atual
        current_best = min(population, key=lambda ind: calculate_distance(ind, points))
        current_best_distance = calculate_distance(current_best, points)
        history.append(current_best_distance)  # salva a melhor distância encontrada nessa geração
        
        # Verifica se houve melhoria
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_solution = current_best
            generations_without_improvement = 0  # reseta o contador de gerações sem melhoria
        else:
            generations_without_improvement += 1  # incrementa o contador

        # Critério de parada por falta de melhoria
        if generations_without_improvement >= patience:
            print(f"Parada antecipada após {generations_without_improvement} gerações sem melhoria.")
            break

        # Mostrando solução intermediária a cada 100 gerações com o nome do cenário
        if generation % 100 == 0:
            print(f"Geração {generation}: Melhor distância = {best_distance:.2f} (Cenário: {scenario_name})")

    return best_solution, best_distance, history

# Executa o algoritmo para os dois cenários
start_time = time.time()
best_uniform, dist_uniform, history_uniform = genetic_algorithm(points_uniform, pop_size, max_generations, initial_mutation_rate, patience, "Pontos Uniformes")
time_uniform = time.time() - start_time

start_time = time.time()
best_circle, dist_circle, history_circle = genetic_algorithm(points_circle, pop_size, max_generations, initial_mutation_rate, patience, "Pontos Circulares")
time_circle = time.time() - start_time

# Exibe os resultados
print(f"Melhor distância para pontos uniformes: {dist_uniform:.2f}, Tempo de execução: {time_uniform:.2f}s")
print(f"Melhor distância para pontos circulares: {dist_circle:.2f}, Tempo de execução: {time_circle:.2f}s")

# Gráfico da evolução da solução ao longo das gerações para Pontos Uniformes
plt.figure(figsize=(12, 6))  # Aumenta o tamanho da figura
plt.plot(history_uniform, label='Pontos Uniformes', linestyle='-', marker='o', markersize=4, color='blue', alpha=0.7)
plt.xlabel('Geração', fontsize=12)  # Melhora a aparência do eixo X
plt.ylabel('Melhor Distância', fontsize=12)  # Melhora a aparência do eixo Y
plt.title('Evolução da Melhor Solução - Pontos Uniformes', fontsize=14, fontweight='bold')  # Título mais destacado
plt.legend(title='Cenário', fontsize=10)  # Adiciona um título à legenda
plt.grid(True, linestyle='--', alpha=0.7)  # Grade mais suave
plt.tight_layout()  # Ajusta automaticamente para que os elementos do gráfico não fiquem cortados
plt.show()

# Gráfico da evolução da solução ao longo das gerações para Pontos Circulares
plt.figure(figsize=(12, 6))  # Aumenta o tamanho da figura
plt.plot(history_circle, label='Pontos Circulares', linestyle='--', marker='s', markersize=4, color='green', alpha=0.7)
plt.xlabel('Geração', fontsize=12)  # Melhora a aparência do eixo X
plt.ylabel('Melhor Distância', fontsize=12)  # Melhora a aparência do eixo Y
plt.title('Evolução da Melhor Solução - Pontos Circulares', fontsize=14, fontweight='bold')  # Título mais destacado
plt.legend(title='Cenário', fontsize=10)  # Adiciona um título à legenda
plt.grid(True, linestyle='--', alpha=0.7)  # Grade mais suave
plt.tight_layout()  # Ajusta automaticamente para que os elementos do gráfico não fiquem cortados
plt.show()

def execute_for_100_points():
    # Definindo os parâmetros do problema
    num_points = 100  # número de pontos
    pop_size = 100  # tamanho da população
    max_generations = 1000  # número máximo de gerações
    initial_mutation_rate = 0.1  # taxa de mutação inicial (10%)
    patience = 200  # número de gerações sem melhoria antes de parar

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points_circle = np.c_[50 + 40 * np.cos(theta), 50 + 40 * np.sin(theta)]  # pontos dispostos em círculo

    print("Executando para 100 pontos circulares...")
    start_time = time.time()
    best_circle, dist_circle, history_circle = genetic_algorithm(points_circle, pop_size, max_generations, initial_mutation_rate, patience, "Pontos Circulares")
    time_circle = time.time() - start_time

    print(f"Melhor distância para 100 pontos circulares: {dist_circle:.2f}, Tempo de execução: {time_circle:.2f}s")

    # Gráfico da evolução da solução ao longo das gerações para Pontos Circulares
    plt.figure(figsize=(12, 6))
    plt.plot(history_circle, label='Pontos Circulares', linestyle='--', marker='s', markersize=4, color='green', alpha=0.7)
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Melhor Distância', fontsize=12)
    plt.title('Evolução da Melhor Solução - 100 Pontos Circulares', fontsize=14, fontweight='bold')
    plt.legend(title='Cenário', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Chama a função para executar o algoritmo com 100 pontos
execute_for_100_points()