import numpy as np

# =========================
# Funções de Penalidade
# =========================

# Penalidade padrão (Pen3 é a mais forte segundo o artigo)
DEFAULT_PENALTY = 3


def get_violated_constraints(x, R, b):
    """
    Retorna o conjunto I de índices das restrições violadas.
    I = {i | sum_j(w_ij * s_j) > c_i}
    """
    usage = np.dot(R, x)
    return np.where(usage > b)[0]


def compute_sum_j(x, R, I):
    """
    Calcula sum_j para cada item j.
    sum_j = soma dos pesos w_ij para todas as restrições violadas i ∈ I
    
    Args:
        x: cromossomo binário
        R: matriz de pesos (m x n)
        I: conjunto de índices de restrições violadas
    
    Retorna:
        array de sum_j para cada j
    """
    n = len(x)
    sum_j = np.zeros(n)
    
    if len(I) > 0:
        # Soma as linhas violadas para cada coluna
        sum_j = np.sum(R[I, :], axis=0)
    
    return sum_j


def penalty_1(x, p, R, b):
    """
    Pen1(S) = 1 / SUM1
    onde SUM1 = sum_{j ∈ J} sum_j
    
    Retorna 0 se solução é viável.
    """
    I = get_violated_constraints(x, R, b)
    
    if len(I) == 0:
        return 1.0
    
    sum_j = compute_sum_j(x, R, I)
    J = np.where(x == 1)[0]  # Itens selecionados
    
    SUM1 = np.sum(sum_j[J])
    
    if SUM1 == 0:
        return 0
    
    return 1.0 / SUM1


def penalty_2(x, p, R, b):
    """
    Pen2(S) = 1 / SUM2
    onde SUM2 = sum_{j ∈ J} (1 / sum_j)
    
    Retorna 0 se solução é viável.
    """
    I = get_violated_constraints(x, R, b)
    
    if len(I) == 0:
        return 1.0
    
    sum_j = compute_sum_j(x, R, I)
    J = np.where(x == 1)[0]
    
    # Evita divisão por zero
    sum_j_selected = sum_j[J]
    valid_mask = sum_j_selected > 0
    
    if not np.any(valid_mask):
        return 0
    
    SUM2 = np.sum(1.0 / sum_j_selected[valid_mask])
    
    if SUM2 == 0:
        return 0
    
    return 1.0 / SUM2


def penalty_3(x, p, R, b):
    """
    Pen3(S) = 1 / sum_{j ∈ J} (p_j * sum_j)
    
    Penalidade mais forte - inclui lucro proporcional.
    Retorna 0 se solução é viável.
    """
    I = get_violated_constraints(x, R, b)
    
    if len(I) == 0:
        return 1.0
    
    sum_j = compute_sum_j(x, R, I)
    J = np.where(x == 1)[0]
    
    # sum_{j ∈ J} (p_j * sum_j)
    weighted_sum = np.sum(p[J] * sum_j[J])
    
    if weighted_sum == 0:
        return 0
    
    return 1.0 / weighted_sum


def penalty_4(x, p, R, b):
    """
    Pen4(S) = 1 / sum_{j ∈ J} (p_j / sum_j)
    
    Retorna 0 se solução é viável.
    """
    I = get_violated_constraints(x, R, b)
    
    if len(I) == 0:
        return 1.0
    
    sum_j = compute_sum_j(x, R, I)
    J = np.where(x == 1)[0]
    
    # Evita divisão por zero
    sum_j_selected = sum_j[J]
    valid_mask = sum_j_selected > 0
    
    if not np.any(valid_mask):
        return 0
    
    # sum_{j ∈ J} (p_j / sum_j) para sum_j > 0
    weighted_sum = np.sum(p[J][valid_mask] / sum_j_selected[valid_mask])
    
    if weighted_sum == 0:
        return 0
    
    return 1.0 / weighted_sum


# Dicionário para selecionar função de penalidade
PENALTY_FUNCTIONS = {
    1: penalty_1,
    2: penalty_2,
    3: penalty_3,  # Recomendada (mais forte)
    4: penalty_4
}


def get_penalty(x, p, R, b, penalty_type=DEFAULT_PENALTY):
    """
    Calcula a penalidade para uma solução.
    
    Args:
        x: solução binária (cromossomo)
        p: vetor de lucros
        R: matriz de restrições
        b: vetor de capacidades
        penalty_type: tipo de penalidade (1, 2, 3 ou 4)
    
    Retorna:
        valor da penalidade (0 se viável)
    """
    penalty_func = PENALTY_FUNCTIONS.get(penalty_type, penalty_3)
    return penalty_func(x, p, R, b)
