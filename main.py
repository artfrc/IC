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
    population = generate_initial_population(p, R, b, pop_size=POPULATION_SIZE)

    # Validação da população
    test_population(population, R, b)

    # Exibe fitness da população inicial
    initial_fitness = [evaluate_solution(x, p) for x in population]
    print(f"Fitness inicial - Melhor: {max(initial_fitness)}, Média: {np.mean(initial_fitness):.2f}")

    # Executa o algoritmo genético
    print(f"\nExecutando algoritmo genético por {GENERATIONS} gerações...")
    best_solution, best_fitness, history = genetic_algorithm(
        population, p, R, b, 
        generations=GENERATIONS, 
        elitism=True
    )

    # Valida a solução final
    assert check_feasibility(best_solution, R, b), "Solução final inviável!"

    # Exibe resultados
    print_ga_results(best_solution, best_fitness, history, optimum, GENERATIONS)


if __name__ == "__main__":
    main()
