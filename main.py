import numpy as np
from init import (
    read_sento1,
    generate_initial_population,
    test_population,
    evaluate_solution,
    check_feasibility
)
from genetic_algorithm import (
    genetic_algorithm,
    print_ga_results,
    MUTATION_RATE,
    M_CROSSOVER_POINTS
)


POPULATION_SIZE = 30
GENERATIONS = 1000


def main():
    # Leitura do dataset
    p, R, b, m, n, optimum = read_sento1("dataset_sento1.txt")

    print(f"{'='*50}")
    print("PROBLEMA DA MOCHILA MULTIDIMENSIONAL")
    print(f"{'='*50}")
    print(f"Instância: SENTO1")
    print(f"Itens: {n}, Restrições: {m}")
    print(f"Ótimo conhecido: {optimum}")
    print(f"\nParâmetros do AG:")
    print(f"  População: {POPULATION_SIZE}")
    print(f"  Gerações: {GENERATIONS}")
    print(f"  Taxa de mutação: {MUTATION_RATE*100:.1f}%")
    print(f"  Pontos de crossover: {M_CROSSOVER_POINTS}")

    # Geração da população inicial
    print(f"\nGerando população inicial...")
    population = generate_initial_population(p, R, b, pop_size=5*n)

    # Executa o algoritmo genético
    print(f"\nExecutando algoritmo genético por {GENERATIONS} gerações...")
    best_solution, best_fitness, history = genetic_algorithm(
        population, p, R, b, 
        generations=GENERATIONS, 
        elitism=True
    )

    # Verifica viabilidade da solução final
    is_feasible = check_feasibility(best_solution, R, b)

    # Exibe resultados
    print_ga_results(best_solution, best_fitness, history, optimum, GENERATIONS, is_feasible, p=p)



if __name__ == "__main__":
    main()
