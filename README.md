# Algoritmo Genético para o Problema da Mochila Multidimensional (MKP)

## O Problema

O **Problema da Mochila Multidimensional (MKP)** consiste em selecionar um subconjunto de itens que maximize o lucro total, respeitando múltiplas restrições de capacidade.

### Formulação Matemática

$$\max \sum_{j=1}^{n} p_j \cdot x_j$$

Sujeito a:
$$\sum_{j=1}^{n} w_{ij} \cdot x_j \leq c_i, \quad \forall i \in \{1, ..., m\}$$
$$x_j \in \{0, 1\}, \quad \forall j \in \{1, ..., n\}$$

## Variáveis

| Variável | Descrição |
|----------|-----------|
| $n$ | Número de itens |
| $m$ | Número de restrições (knapsack)|
| $p_j$ | Lucro do item $j$ |
| $w_{ij}$ | Peso do item $j$ na restrição $i$ |
| $c_i$ | Capacidade da restrição $i$ |
| $x_j$ | Variável binária (1 = item selecionado) |

## Algoritmo Genético

### Representação
Cada indivíduo é um vetor binário de tamanho $n$, onde `x[j] = 1` indica que o item $j$ está na mochila.

### Componentes

| Componente | Método |
|------------|--------|
| **População Inicial** | Heurística de Dantzig com restrição substituta |
| **Fitness** | $f(x) = \sum_{j=1}^{n} p_j \cdot x_j$ |
| **Seleção** | Stochastic Universal Sampling (SUS) |
| **Crossover** | M-point crossover (M=3) |
| **Mutação** | Bit-flip com taxa de 5% |
| **Reparo** | Remove itens até viabilidade + greedy fill |

### Parâmetros

```python
POPULATION_SIZE = 30
GENERATIONS = 100
MUTATION_RATE = 0.05
M_CROSSOVER_POINTS = 3
```

## Execução

```bash
python main.py
```

## Estrutura

```
├── main.py              # Ponto de entrada
├── init.py              # Leitura de dados e geração inicial
├── genetic_algorithm.py # Implementação do AG
└── dataset_sento1.txt   # Instância SENTO1
```