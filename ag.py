import numpy as np

# =========================
# Leitura do dataset SENTO1
# =========================

def read_sento1(filename):
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


# =====================================
# Dantzig com restrição substituta
# =====================================

def dantzig_surrogate(p, R, b, lambdas):
    w = np.dot(lambdas, R)
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


# =========================
# Repair de factibilidade
# =========================

def repair_solution(x, R, b):
    while True:
        violations = np.dot(R, x) - b
        if np.all(violations <= 0):
            break

        i = np.argmax(violations)
        items = np.where(x == 1)[0]

        j = items[np.argmax(R[i, items])]
        x[j] = 0

    return x


# =========================
# Fill viável
# =========================

def greedy_fill(x, p, R, b):
    used = np.dot(R, x)
    free_items = np.where(x == 0)[0]

    scores = [(p[j] / (np.sum(R[:, j]) + 1e-6), j) for j in free_items]
    scores.sort(reverse=True)

    for _, j in scores:
        if np.all(used + R[:, j] <= b):
            x[j] = 1
            used += R[:, j]

    return x


# =================================
# Gerador de população inicial
# =================================

def generate_initial_population(p, R, b, pop_size=30):
    m = R.shape[0]
    population = []

    for _ in range(pop_size):
        lambdas = (1 / b) * np.random.uniform(0.8, 1.2, size=m)

        x = dantzig_surrogate(p, R, b, lambdas)
        x = repair_solution(x, R, b)
        x = greedy_fill(x, p, R, b)

        population.append(x)

    return population   # 🔴 ISSO FALTAVA


# =========================
# Testes
# =========================

def check_feasibility(x, R, b):
    return np.all(np.dot(R, x) <= b)


def test_population(population, R, b):
    for i, x in enumerate(population):
        assert set(np.unique(x)).issubset({0, 1}), f"Indivíduo {i} não binário"
        assert check_feasibility(x, R, b), f"Indivíduo {i} inviável"
    print("✔️ Todos os indivíduos são binários e viáveis")


# =========================
# Execução  
# =========================

if __name__ == "__main__":
    p, R, b, m, n, optimum = read_sento1("dataset_sento1.txt")

    population = generate_initial_population(p, R, b, pop_size=30)

    test_population(population, R, b)

    print(f"\nInstância SENTO1")
    print(f"Itens: {n}, Restrições: {m}")
    print(f"Ótimo conhecido: {optimum}\n")

    for i, x in enumerate(population[:5]):
        value = np.dot(p, x)
        print(f"Indivíduo {i+1}: valor = {value}")
