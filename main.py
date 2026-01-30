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
    M_CROSSOVER_POINTS,
    print_history
)



def main():
    p, R, b, m, n, optimum = read_sento1("dataset_sento1.txt")

    GENERATIONS = 3000
    POP_SIZE = 5*n
    CROSSOVER_RATE = 0.95  # Taxa de crossover (95%)

    print(f"{'='*50}")
    print("PROBLEMA DA MOCHILA MULTIDIMENSIONAL")
    print(f"{'='*50}")
    print(f"Instância: SENTO1")
    print(f"Itens: {n}, Restrições: {m}")
    print(f"Ótimo conhecido: {optimum}")
    print(f"\nParâmetros do AG:")
    print(f"  População: {POP_SIZE}")
    print(f"  Gerações: {GENERATIONS}")
    print(f"  Pontos de crossover: {M_CROSSOVER_POINTS}")

    # Geração da população inicial
    print(f"\nGerando população inicial...")
    population = generate_initial_population(p, R, b, pop_size=POP_SIZE)

    # Executa o algoritmo genético
    print(f"\nExecutando algoritmo genético por {GENERATIONS} gerações...")
    best_solution, best_fitness, history = genetic_algorithm(
        population, p, R, b,
        generations=GENERATIONS,
        elitism=True,
        crossover_rate=CROSSOVER_RATE
    )

    # Verifica viabilidade da solução final
    is_feasible = check_feasibility(best_solution, R, b)

    # Exibe resultados
    print_ga_results(best_solution, best_fitness, history, optimum, GENERATIONS, is_feasible, p=p)
    #print_history(history)



if __name__ == "__main__":
    main()
