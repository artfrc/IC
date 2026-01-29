import numpy as np

def read_sento1(filename):
    """
    Lê um arquivo no formato SENTO1 para o problema da mochila multidimensional.
    
    Retorna:
        p: vetor de lucros
        R: matriz de restrições (m x n)
        b: vetor de capacidades
        m: número de restrições
        n: número de itens
        optimum: valor ótimo conhecido
    """
    with open(filename, 'r') as f:
        data = [int(x) for x in f.read().split()]

    idx = 0

    m = data[idx]; idx += 1  # restrições
    n = data[idx]; idx += 1  # itens

    p = np.array(data[idx:idx + n])
    idx += n

    b = np.array(data[idx:idx + m])
    idx += m

    R = np.zeros((m, n), dtype=int)
    for i in range(m):
        R[i, :] = data[idx:idx + n]
        idx += n

    optimum = data[idx]

    return p, R, b, m, n, optimum



def dantzig_surrogate(p, R, b, lambdas):
    """
    Aplica a heurística de Dantzig com restrição substituta.
    
    Args:
        p: vetor de lucros
        R: matriz de restrições
        b: vetor de capacidades
        lambdas: multiplicadores para a restrição substituta
    
    Retorna:
        x: solução binária (pode ser inviável)
    """
    w = np.dot(lambdas, R)
    w[w == 0] = 1e-9
    rho = p / w

    order = np.argsort(-rho)

    x = np.zeros(len(p), dtype=int)
    capacity = np.sum(lambdas * b)
    used = 0.0

    for j in order:
        if used + w[j] <= capacity:
            x[j] = 1
            used += w[j]

    return x



def repair_solution(x, R, b):
    """
    Repara uma solução inviável removendo itens até que todas as
    restrições sejam satisfeitas.
    
    Args:
        x: solução binária (possivelmente inviável)
        R: matriz de restrições
        b: vetor de capacidades
    
    Retorna:
        x: solução binária viável
    """
    while True:
        violations = np.dot(R, x) - b
        if np.all(violations <= 0):
            break

        i = np.argmax(violations)
        items = np.where(x == 1)[0]

        j = items[np.argmax(R[i, items])]
        x[j] = 0

    return x



def greedy_fill(x, p, R, b):
    """
    Preenche uma solução viável com itens adicionais de forma gulosa.
    
    Args:
        x: solução binária viável
        p: vetor de lucros
        R: matriz de restrições
        b: vetor de capacidades
    
    Retorna:
        x: solução binária viável melhorada
    """
    used = np.dot(R, x)
    free_items = np.where(x == 0)[0]

    scores = [(p[j] / (np.sum(R[:, j]) + 1e-6), j) for j in free_items]
    scores.sort(reverse=True)

    for _, j in scores:
        if np.all(used + R[:, j] <= b):
            x[j] = 1
            used += R[:, j]

    return x



def generate_initial_population(p, R, b, pop_size=30):
    """
    Gera uma população inicial de soluções para o algoritmo genético.
    As soluções podem ser inviáveis (serão penalizadas na função fitness).
    
    Args:
        p: vetor de lucros
        R: matriz de restrições
        b: vetor de capacidades
        pop_size: tamanho da população
    
    Retorna:
        population: lista de soluções binárias (possivelmente inviáveis)
    """
    m = R.shape[0]
    population = []

    for _ in range(pop_size):
        lambdas = (1 / b) * np.random.uniform(0.8, 1.2, size=m)

        x = dantzig_surrogate(p, R, b, lambdas)
        # Soluções não são reparadas - podem ser inviáveis

        population.append(x)

    return np.array(population)



def check_feasibility(x, R, b):
    """Verifica se uma solução é viável."""
    return np.all(np.dot(R, x) <= b)


def test_population(population, R, b):
    """Testa se todos os indivíduos da população são válidos."""
    for i, x in enumerate(population):
        assert set(np.unique(x)).issubset({0, 1}), f"Indivíduo {i} não binário"
    print("✔️ Todos os indivíduos são binários e viáveis")


def evaluate_solution(x, p):
    """Calcula o valor (lucro) de uma solução."""
    return np.dot(p, x)
