import numpy as np
from init import (
    repair_solution,
    greedy_fill,
    check_feasibility
)

# =========================
# Constantes do AG
# =========================

MUTATION_RATE = 0.05  # Taxa de mutação (5%)
M_CROSSOVER_POINTS = 3  # Número de pontos de crossover


# =========================
# Função de Fitness
# =========================

def fitness(x, p):
    """
    Calcula o fitness de um indivíduo.
    Soma os lucros dos itens selecionados (genes = 1).
    
    Args:
        x: solução binária (cromossomo)
        p: vetor de lucros
    
    Retorna:
        valor total do lucro
    """
    return np.dot(p, x)



def stochastic_universal_sampling(population, fitness_values, num_parents):
    """
    Stochastic Universal Sampling (SUS) para seleção de pais.
    
    Args:
        population: lista de indivíduos
        fitness_values: lista de valores de fitness
        num_parents: número de pais a selecionar
    
    Retorna:
        lista de pais selecionados
    """
    total_fitness = np.sum(fitness_values)
    
    # Evita divisão por zero
    if total_fitness == 0:
        return [population[i].copy() for i in np.random.choice(len(population), num_parents)]
    
    # Distância entre ponteiros
    pointer_distance = total_fitness / num_parents
    
    # Ponto de início aleatório
    start_point = np.random.uniform(0, pointer_distance)
    
    # Posições dos ponteiros
    pointers = [start_point + i * pointer_distance for i in range(num_parents)]
    
    # Seleção dos pais usando a roleta
    selected_parents = []
    cumulative_fitness = np.cumsum(fitness_values)
    
    for pointer in pointers:
        for i, cum_fit in enumerate(cumulative_fitness):
            if pointer <= cum_fit:
                selected_parents.append(population[i].copy())
                break
    
    return selected_parents


def m_point_crossover(parent1, parent2, m=M_CROSSOVER_POINTS):
    """
    M-point crossover entre dois pais.
    
    Cria dois filhos:
    - Filho 1: segmentos pares do pai 1, ímpares do pai 2
    - Filho 2: segmentos ímpares do pai 1, pares do pai 2
    
    Args:
        parent1: primeiro pai (array binário)
        parent2: segundo pai (array binário)
        m: número de pontos de crossover
    
    Retorna:
        child1, child2: dois filhos gerados
    """
    n = len(parent1)
    
    # Gera M pontos de crossover únicos e ordenados
    crossover_points = sorted(np.random.choice(range(1, n), size=min(m, n-1), replace=False))
    
    # Adiciona início e fim para facilitar a segmentação
    points = [0] + list(crossover_points) + [n]
    
    child1 = np.zeros(n, dtype=int)
    child2 = np.zeros(n, dtype=int)
    
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        if i % 2 == 0:  # Segmento par
            child1[start:end] = parent1[start:end]
            child2[start:end] = parent2[start:end]
        else:  # Segmento ímpar
            child1[start:end] = parent2[start:end]
            child2[start:end] = parent1[start:end]
    
    return child1, child2


def crossover_and_select_best(parent1, parent2, p, R, b, m=M_CROSSOVER_POINTS):
    """
    Realiza crossover, repara os filhos e retorna o melhor.
    
    Args:
        parent1, parent2: pais
        p: vetor de lucros
        R: matriz de restrições
        b: vetor de capacidades
        m: número de pontos de crossover
    
    Retorna:
        melhor filho (viável)
    """
    child1, child2 = m_point_crossover(parent1, parent2, m)
    
    # Repara e preenche os filhos para garantir viabilidade
    child1 = repair_solution(child1, R, b)
    child1 = greedy_fill(child1, p, R, b)
    
    child2 = repair_solution(child2, R, b)
    child2 = greedy_fill(child2, p, R, b)
    
    # Retorna o filho com melhor fitness
    if fitness(child1, p) >= fitness(child2, p):
        return child1
    else:
        return child2



def mutate(x, mutation_rate=MUTATION_RATE):
    """
    Aplica mutação bit-flip com probabilidade mutation_rate.
    
    Args:
        x: indivíduo (array binário)
        mutation_rate: probabilidade de mutação por gene
    
    Retorna:
        indivíduo mutado
    """
    mutated = x.copy()
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]  # Flip do bit
    
    return mutated



def genetic_algorithm(population, p, R, b, generations=100, elitism=True):
    """
    Executa o algoritmo genético.
    
    Args:
        population: população inicial
        p: vetor de lucros
        R: matriz de restrições
        b: vetor de capacidades
        generations: número de gerações
        elitism: se True, mantém o melhor indivíduo
    
    Retorna:
        best_solution: melhor solução encontrada
        best_fitness: fitness da melhor solução
        history: histórico de fitness por geração
    """
    pop_size = len(population)
    current_pop = [ind.copy() for ind in population]
    
    # Histórico para acompanhamento
    history = {
        'best': [],
        'average': [],
        'worst': []
    }
    
    best_ever = None
    best_ever_fitness = -1
    
    for gen in range(generations):
        # Calcula fitness de todos os indivíduos
        fitness_values = np.array([fitness(ind, p) for ind in current_pop])
        
        # Atualiza melhor solução global
        gen_best_idx = np.argmax(fitness_values)
        if fitness_values[gen_best_idx] > best_ever_fitness:
            best_ever = current_pop[gen_best_idx].copy()
            best_ever_fitness = fitness_values[gen_best_idx]
        
        # Registra histórico
        history['best'].append(np.max(fitness_values))
        history['average'].append(np.mean(fitness_values))
        history['worst'].append(np.min(fitness_values))
        
        # Seleção de pais via SUS
        num_parents = pop_size
        parents = stochastic_universal_sampling(current_pop, fitness_values, num_parents)
        
        # Gera nova população através de crossover
        new_population = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Crossover e seleção do melhor filho
            child = crossover_and_select_best(parent1, parent2, p, R, b)
            
            # Mutação
            child = mutate(child)
            
            # Repara após mutação para garantir viabilidade
            child = repair_solution(child, R, b)
            child = greedy_fill(child, p, R, b)
            
            new_population.append(child)
        
        # Se população tem tamanho ímpar, adiciona mais um indivíduo
        while len(new_population) < pop_size:
            if elitism:
                new_population.append(best_ever.copy())
            else:
                idx = np.random.randint(len(parents))
                new_population.append(parents[idx].copy())
        
        # Elitismo: substitui o pior pelo melhor de todos os tempos
        if elitism and best_ever is not None:
            new_fitness = [fitness(ind, p) for ind in new_population]
            worst_idx = np.argmin(new_fitness)
            new_population[worst_idx] = best_ever.copy()
        
        current_pop = new_population
    
    return best_ever, best_ever_fitness, history


def print_ga_results(best_solution, best_fitness, history, optimum, generations):
    """
    Exibe os resultados do algoritmo genético.
    """
    print(f"\n{'='*50}")
    print("RESULTADOS DO ALGORITMO GENÉTICO")
    print(f"{'='*50}")
    print(f"Melhor fitness encontrado: {best_fitness}")
    print(f"Ótimo conhecido: {optimum}")
    print(f"Gap para o ótimo: {((optimum - best_fitness) / optimum) * 100:.2f}%")
    print(f"Itens selecionados: {np.sum(best_solution)}")
    print(f"\nEvolução do fitness:")
    print(f"  Geração 1   - Melhor: {history['best'][0]:.0f}, Média: {history['average'][0]:.2f}")
    print(f"  Geração {generations} - Melhor: {history['best'][-1]:.0f}, Média: {history['average'][-1]:.2f}")
