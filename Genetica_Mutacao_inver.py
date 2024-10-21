import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import time

# Definindo os parâmetros do problema
num_points = 10  # número de pontos
pop_size = 100  # tamanho da população
max_generations = 1000  # número máximo de gerações
initial_mutation_rate = 0.1  # taxa de mutação inicial
patience = 200  # gerações sem melhoria antes de parar

# Gerando pontos aleatórios para os cenários
np.random.seed(42)
points_uniform = np.random.rand(num_points, 2) * 100  # pontos distribuídos uniformemente
theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
points_circle = np.c_[50 + 40 * np.cos(theta), 50 + 40 * np.sin(theta)]  # pontos em círculo

# Função auxiliar para calcular a distância total de um caminho
def calculate_distance(path, points):
    return sum(euclidean(points[path[i]], points[path[i + 1]]) for i in range(len(path) - 1)) + \
           euclidean(points[path[-1]], points[path[0]])

# Função de aptidão (quanto menor a distância, maior a aptidão)
def fitness(path, points):
    return 1 / calculate_distance(path, points)

# Criação da população inicial (sequências aleatórias de pontos)
def create_initial_population(pop_size, num_points):
    return [random.sample(range(num_points), num_points) for _ in range(pop_size)]

# Seleção (torneio)
def tournament_selection(population, points, k=5):
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: fitness(ind, points), reverse=True)
    return selected[0]

# Cruzamento do tipo Order Crossover (OX)
def order_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b] = parent1[a:b]

    fill_pos = b
    for elem in parent2:
        if elem not in child:
            if fill_pos >= size:
                fill_pos = 0
            child[fill_pos] = elem
            fill_pos += 1
    return child

def inversion_mutation(path):
    # Escolhe dois índices aleatórios para definir os limites da inversão
    i, j = sorted(random.sample(range(len(path)), 2))
    path[i:j+1] = reversed(path[i:j+1])  # inverte a subsequência entre os dois índices
    return path

# Modificando a função para usar essa nova mutação em vez da original
def mutate(path, mutation_rate, mutation_type="inversion"):
    if random.random() < mutation_rate:  # aplica mutação com uma certa chance
        if mutation_type == "inversion":
            return inversion_mutation(path)
    return path

# Executando o Algoritmo Genético com a nova função de mutação
def genetic_algorithm(points, pop_size, max_generations, mutation_rate, patience, scenario_name=""):
    # Criação da população inicial
    population = create_initial_population(pop_size, len(points))
    best_solution = None
    best_distance = float('inf')  # inicializa com uma distância muito grande
    history = []  # para armazenar a evolução da solução ao longo das gerações
    generations_without_improvement = 0  # contador de gerações sem melhoria

    # Iterando ao longo das gerações
    for generation in range(max_generations):
        
        new_population = []  # nova geração
        for _ in range(pop_size // 2):
            # Seleciona os pais e faz cruzamento
            parent1 = tournament_selection(population, points)
            parent2 = tournament_selection(population, points)
            child1 = order_crossover(parent1, parent2)
            child2 = order_crossover(parent2, parent1)
            # Adiciona os filhos mutados à nova população usando a mutação de inversão
            new_population.extend([mutate(child1, mutation_rate, "inversion"), mutate(child2, mutation_rate, "inversion")])
        
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

# Executa o algoritmo para os dois cenários com a nova mutação
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

'''
1. População e Critério de Parada 

População: Foi escolhida uma população de 100 indivíduos para manter um equilíbrio entre diversidade
e capacidade de explorar boas soluções. Uma população maior poderia aumentar a diversidade, mas ao 
custo de mais tempo de processamento. Populações menores podem convergir rapidamente, mas podem levar a 
soluções subótimas, já que podem explorar menos o espaço de soluções.

Critério de Parada: O critério de parada combina duas abordagens:
1- Um número máximo de gerações (1000), para limitar o tempo de execução.
2- Um critério de estagnação (patience = 200 gerações sem melhoria). Se não houver melhoria por 200 gerações 
consecutivas, o algoritmo para.

2. Taxa de Mutação

A taxa de mutação foi definida como 10% (0.1). A escolha desse valor visa manter um equilíbrio entre a exploração 
e a preservação de boas soluções

3. Representação do Gene e Cruzamento

Representação do Gene: A representação dos genes é uma lista de índices que indicam a ordem em que os pontos (cidades
no caso do TSP) devem ser visitados. Isso é apropriado para problemas de ordenação como o TSP, onde a sequência dos genes
é crucial para determinar a qualidade da solução.
Cruzamento (Order Crossover - OX): O cruzamento utilizado é o Order Crossover (OX), ideal para problemas de ordenação, pois 
garante que a ordem relativa dos elementos seja preservada. Ele combina duas soluções existentes de maneira eficiente, sem 
introduzir duplicatas nem omitir genes, mantendo assim a validade das soluções.

4. Função de Aptidão (fitness)
A função de aptidão é baseada no inverso da distância total percorrida no caminho (rotação entre os pontos). Como o objetivo 
do problema é minimizar a distância, a aptidão é inversamente proporcional à distância: soluções com distâncias menores terão 
aptidões maiores.


'''