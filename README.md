# 🧬 Algoritmo Genético Híbrido para o Problema da Mochila Multidimensional (MKP)

Este projeto implementa um **Algoritmo Genético Híbrido (HGA)** para resolver o **Problema da Mochila Multidimensional (Multidimensional Knapsack Problem – MKP)**, inspirado no artigo clássico que combina a heurística de Dantzig com funções de penalidade, em especial a penalidade **Pen3**.

A instância utilizada para validação foi a **SENTO1 (OR-Library)**, contendo **60 itens** e **30 restrições**, com ótimo conhecido igual a **7772**.

O algoritmo implementado é capaz de atingir exatamente o ótimo global dessa instância.

---

## 📌 Visão Geral

O MKP consiste em selecionar um subconjunto de itens de modo a maximizar o lucro total, respeitando múltiplas restrições de capacidade:

\[
\max \sum_{j=1}^{n} p_j x_j
\]

sujeito a:

\[
\sum_{j=1}^{n} w_{ij} x_j \le b_i, \quad i = 1,\dots,m
\]

onde:

- \(p_j\): lucro do item \(j\)  
- \(w_{ij}\): consumo do item \(j\) na restrição \(i\)  
- \(b_i\): capacidade da restrição \(i\)  
- \(x_j \in \{0,1\}\): decisão de selecionar ou não o item  

---

## 📂 Estrutura do Projeto


## Execução

```bash
python main.py
```

## 📂 Estrutura do Projeto

```
├── main.py              # Ponto de entrada
├── init.py              # Leitura de dados e geração inicial
├── genetic_algorithm.py # Implementação do AG
└── dataset_sento1.txt   # Instância SENTO1
```

---

## ⚙️ Principais Modificações Implementadas

### 1. Leitura do Dataset (SENTO1)

Foi implementada a função `read_sento1`, responsável por:

- Ler o arquivo da OR-Library;
- Extrair:
  - número de restrições (`m`);
  - número de itens (`n`);
  - vetor de lucros (`p`);
  - vetor de capacidades (`b`);
  - matriz de consumo (`R`);
  - ótimo conhecido.

Isso garante a reconstrução correta do modelo MKP diretamente a partir do arquivo texto.

---

### 2. População Inicial com Dantzig + Restrição Substituta

A população inicial é gerada utilizando a heurística de **Dantzig com restrição substituta**:

- Combinação linear das restrições por multiplicadores aleatórios;
- Ordenação dos itens pela razão lucro/peso;
- Inserção gulosa enquanto a capacidade substituta permite.

Além disso, foi adicionada proteção contra divisão por zero:

```python
w[w == 0] = 1e-9