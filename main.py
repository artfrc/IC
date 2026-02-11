import numpy as np
import time
from statistics import median
from init import (
    read_file,
    generate_initial_population,
    check_feasibility
)
from genetic_algorithm import (
    genetic_algorithm,
    print_ga_results
)


def main():
    p, R, b, m, n, optimum = read_file("dataset_weish23.txt")

    GENERATIONS = 3000
    POP_SIZE = 5 * n
    CROSSOVER_RATE = 0.95

    TARGET = optimum          # ótimo do artigo
    MAX_RUNS = 10

    results = []              # fitness de cada run
    times = []                # tempo de cada run
    gaps = []                 # GAP de cada run

    best_global_fitness = -1
    best_global_solution = None
    best_run_index = -1

    run = 0

    print(f"\n🔬 EXPERIMENTO GA")
    print(f"Ótimo do artigo: {TARGET}")
    print(f"População: {POP_SIZE}, Gerações: {GENERATIONS}\n")

    while run < MAX_RUNS:
        run += 1
        print(f"\n===== RUN {run} =====")

        start_time = time.perf_counter()

        population = generate_initial_population(p, R, b, pop_size=POP_SIZE)

        best_solution, best_fitness, history = genetic_algorithm(
            population, p, R, b,
            generations=GENERATIONS,
            elitism=True,
            crossover_rate=CROSSOVER_RATE
        )

        elapsed_time = time.perf_counter() - start_time

        feasible = check_feasibility(best_solution, R, b)

        gap = ((TARGET - best_fitness) / TARGET) * 100 if TARGET != 0 else 0.0

        results.append(best_fitness)
        times.append(elapsed_time)
        gaps.append(gap)

        print(f"Melhor fitness da run {run}: {best_fitness}")
        print(f"Viável: {feasible}")
        print(f"Tempo da run: {elapsed_time:.2f} s")
        print(f"GAP da run: {gap:.4f} %")

        if best_fitness > best_global_fitness and feasible:
            best_global_fitness = best_fitness
            best_global_solution = best_solution
            best_run_index = run

        # critério de parada: melhor que o artigo
        if best_global_fitness > TARGET:
            print("\n🚀 RESULTADO MELHOR QUE O ARTIGO ENCONTRADO!")
            break

    # ==========================
    # RESULTADOS FINAIS
    # ==========================

    print(f"\n{'='*70}")
    print("📊 RESULTADOS FINAIS DO EXPERIMENTO")
    print(f"{'='*70}")
    print(f"Total de execuções: {run}")
    print(f"Melhor fitness encontrado: {best_global_fitness}")
    print(f"Pior fitness encontrado: {min(results)}")
    print(f"Mediana dos fitness: {median(results)}")
    print(f"Run do melhor resultado: {best_run_index}")

    print(f"\n⏱️ TEMPO DE EXECUÇÃO (segundos)")
    print(f"Melhor tempo: {min(times):.2f}")
    print(f"Pior tempo: {max(times):.2f}")
    print(f"Tempo médio: {np.mean(times):.2f}")

    print(f"\n📉 GAP")
    print(f"Menor GAP: {min(gaps):.4f} %")
    print(f"Maior GAP: {max(gaps):.4f} %")
    print(f"GAP médio: {np.mean(gaps):.4f} %")
    print(f"GAP do melhor resultado: {(TARGET - best_global_fitness)/TARGET * 100:.4f} %")

    print(f"\n🔎 Solução ótima viável: {check_feasibility(best_global_solution, R, b)}")


if __name__ == "__main__":
    main()
