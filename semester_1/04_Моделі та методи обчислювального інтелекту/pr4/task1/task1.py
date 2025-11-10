"""
Лабораторна робота №4
Розв'язання оптимізаційних задач

Генетичний алгоритм для знаходження мінімуму функції Schaffer N.2
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Callable


def schaffer_function_n2(x: float, y: float) -> float:
    """
    Функція Schaffer N.2
    
    Формула: f(x, y) = 0.5 + (sin²(x² - y²) - 0.5) / (1 + 0.001(x² + y²))²
    
    Глобальний мінімум: f(0, 0) = 0
    
    Args:
        x: координата x
        y: координата y
        
    Returns:
        Значення функції в точці (x, y)
    """
    numerator = math.sin(x**2 - y**2)**2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + numerator / denominator


class GeneticAlgorithm:
    """Генетичний алгоритм для оптимізації функцій"""
    
    def __init__(
        self,
        objective_func: Callable[[float, float], float],
        bounds: Tuple[float, float] = (-100, 100),
        population_size: int = 100,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        tournament_size: int = 3,
        elitism_count: int = 2
    ):
        """
        Ініціалізація генетичного алгоритму
        
        Args:
            objective_func: Цільова функція для мінімізації
            bounds: Межі пошуку (min, max)
            population_size: Розмір популяції
            generations: Кількість поколінь
            crossover_rate: Ймовірність схрещування
            mutation_rate: Ймовірність мутації
            mutation_strength: Сила мутації
            tournament_size: Розмір турніру для селекції
            elitism_count: Кількість найкращих особин для елітизму
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.history = []
        
    def generate_population(self) -> List[Tuple[float, float]]:
        """
        Генерація початкової популяції
        
        Returns:
            Список особин (x, y)
        """
        return [
            (
                random.uniform(self.bounds[0], self.bounds[1]),
                random.uniform(self.bounds[0], self.bounds[1])
            )
            for _ in range(self.population_size)
        ]
    
    def evaluate_fitness(self, individual: Tuple[float, float]) -> float:
        """
        Обчислення придатності особини
        
        Для мінімізації: чим менше значення функції, тим більша придатність
        
        Args:
            individual: Особина (x, y)
            
        Returns:
            Значення придатності
        """
        x, y = individual
        value = self.objective_func(x, y)
        # Інвертуємо для мінімізації: чим менше значення, тим більша придатність
        return 1.0 / (1.0 + value)
    
    def tournament_selection(
        self, 
        population: List[Tuple[float, float]], 
        fitnesses: List[float]
    ) -> Tuple[float, float]:
        """
        Турнірна селекція
        
        Args:
            population: Популяція
            fitnesses: Значення придатності
            
        Returns:
            Вибрана особина
        """
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def crossover(
        self, 
        parent1: Tuple[float, float], 
        parent2: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Арифметичне схрещування
        
        Args:
            parent1: Перший батько
            parent2: Другий батько
            
        Returns:
            Два нащадки
        """
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child1 = (
                alpha * parent1[0] + (1 - alpha) * parent2[0],
                alpha * parent1[1] + (1 - alpha) * parent2[1]
            )
            child2 = (
                (1 - alpha) * parent1[0] + alpha * parent2[0],
                (1 - alpha) * parent1[1] + alpha * parent2[1]
            )
            return child1, child2
        return parent1, parent2
    
    def mutate(self, individual: Tuple[float, float]) -> Tuple[float, float]:
        """
        Гауссова мутація
        
        Args:
            individual: Особина для мутації
            
        Returns:
            Мутаційна особина
        """
        x, y = individual
        if random.random() < self.mutation_rate:
            x += random.gauss(0, self.mutation_strength * (self.bounds[1] - self.bounds[0]))
            x = max(self.bounds[0], min(self.bounds[1], x))
        if random.random() < self.mutation_rate:
            y += random.gauss(0, self.mutation_strength * (self.bounds[1] - self.bounds[0]))
            y = max(self.bounds[0], min(self.bounds[1], y))
        return (x, y)
    
    def run(self) -> Tuple[Tuple[float, float], float]:
        """
        Запуск генетичного алгоритму
        
        Returns:
            Кортеж (найкраща особина, найкраще значення функції)
        """
        # Генерація початкової популяції
        population = self.generate_population()
        
        for generation in range(self.generations):
            # Оцінка придатності
            fitnesses = [self.evaluate_fitness(ind) for ind in population]
            
            # Знаходження найкращої особини
            best_idx = np.argmax(fitnesses)
            best_individual = population[best_idx]
            best_value = self.objective_func(best_individual[0], best_individual[1])
            
            # Збереження історії
            self.history.append({
                'generation': generation,
                'best_individual': best_individual,
                'best_value': best_value,
                'avg_fitness': np.mean(fitnesses)
            })
            
            # Виведення прогресу
            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"Покоління {generation:3d}: "
                      f"Найкраще значення = {best_value:.6f} "
                      f"в точці ({best_individual[0]:.4f}, {best_individual[1]:.4f})")
            
            # Елітизм: збереження найкращих особин
            elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
            elite = [population[i] for i in elite_indices]
            
            # Створення нової популяції
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # Селекція
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Схрещування
                child1, child2 = self.crossover(parent1, parent2)
                
                # Мутація
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        # Повернення найкращого результату
        final_fitnesses = [self.evaluate_fitness(ind) for ind in population]
        best_idx = np.argmax(final_fitnesses)
        best_individual = population[best_idx]
        best_value = self.objective_func(best_individual[0], best_individual[1])
        
        return best_individual, best_value


def plot_function_and_solution(
    objective_func: Callable[[float, float], float],
    solution: Tuple[float, float],
    bounds: Tuple[float, float] = (-100, 100),
    resolution: int = 400
):
    """
    Побудова графіка функції та відмітка точки мінімуму
    
    Args:
        objective_func: Цільова функція
        solution: Знайдене рішення (x, y)
        bounds: Межі для побудови графіка
        resolution: Роздільна здатність сітки
    """
    # Створення сітки значень
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Обчислення значень функції (векторизовано для швидкості)
    # Використовуємо numpy для обчислення функції Schaffer N.2
    numerator = np.sin(X**2 - Y**2)**2 - 0.5
    denominator = (1 + 0.001 * (X**2 + Y**2))**2
    Z = 0.5 + numerator / denominator
    
    # Створення фігури з двома підграфіками
    fig = plt.figure(figsize=(16, 6))
    
    # 2D контурний графік
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax1.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax1, label='Значення функції')
    ax1.plot(solution[0], solution[1], 'r*', markersize=20, 
             label=f'Знайдений мінімум\n({solution[0]:.4f}, {solution[1]:.4f})',
             markeredgecolor='white', markeredgewidth=2)
    ax1.plot(0, 0, 'yo', markersize=10, label='Глобальний мінімум (0, 0)',
             markeredgecolor='black', markeredgewidth=1)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Функція Schaffer N.2 (2D вид)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 3D поверхня
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True)
    ax2.scatter([solution[0]], [solution[1]], 
               [objective_func(solution[0], solution[1])],
               color='red', s=200, marker='*', 
               label='Знайдений мінімум', edgecolors='white', linewidths=2)
    ax2.scatter([0], [0], [objective_func(0, 0)],
               color='yellow', s=100, marker='o',
               label='Глобальний мінімум', edgecolors='black', linewidths=1)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('f(x, y)', fontsize=12)
    ax2.set_title('Функція Schaffer N.2 (3D вид)', fontsize=14, fontweight='bold')
    plt.colorbar(surf, ax=ax2, label='Значення функції', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('schaffer_function_optimization.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено у файл 'schaffer_function_optimization.png'")
    plt.show()


def plot_convergence(history: List[dict]):
    """
    Побудова графіка збіжності алгоритму
    
    Args:
        history: Історія виконання алгоритму
    """
    generations = [h['generation'] for h in history]
    best_values = [h['best_value'] for h in history]
    avg_fitnesses = [h['avg_fitness'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Графік найкращих значень
    ax1.plot(generations, best_values, 'b-', linewidth=2, label='Найкраще значення')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Глобальний мінімум (0)')
    ax1.set_xlabel('Покоління', fontsize=12)
    ax1.set_ylabel('Значення функції', fontsize=12)
    ax1.set_title('Збіжність алгоритму', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Графік середньої придатності
    ax2.plot(generations, avg_fitnesses, 'g-', linewidth=2, label='Середня придатність')
    ax2.set_xlabel('Покоління', fontsize=12)
    ax2.set_ylabel('Середня придатність', fontsize=12)
    ax2.set_title('Еволюція придатності популяції', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
    print("Графік збіжності збережено у файл 'convergence.png'")
    plt.show()


def main():
    """Головна функція для запуску оптимізації"""
    print("=" * 70)
    print("Генетичний алгоритм для мінімізації функції Schaffer N.2")
    print("=" * 70)
    
    # Параметри алгоритму
    bounds = (-100, 100)
    population_size = 100
    generations = 200
    
    # Створення та запуск генетичного алгоритму
    ga = GeneticAlgorithm(
        objective_func=schaffer_function_n2,
        bounds=bounds,
        population_size=population_size,
        generations=generations,
        crossover_rate=0.8,
        mutation_rate=0.1,
        mutation_strength=0.05,
        tournament_size=3,
        elitism_count=2
    )
    
    print(f"\nПараметри алгоритму:")
    print(f"  Розмір популяції: {population_size}")
    print(f"  Кількість поколінь: {generations}")
    print(f"  Межі пошуку: [{bounds[0]}, {bounds[1]}]")
    print(f"\nПочаток оптимізації...\n")
    
    # Запуск алгоритму
    best_solution, best_value = ga.run()
    
    # Виведення результатів
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ")
    print("=" * 70)
    print(f"Знайдена точка мінімуму: ({best_solution[0]:.8f}, {best_solution[1]:.8f})")
    print(f"Значення функції: {best_value:.10f}")
    print(f"Відстань від глобального мінімуму (0, 0): "
          f"{math.sqrt(best_solution[0]**2 + best_solution[1]**2):.8f}")
    print(f"Помилка: {abs(best_value - 0.0):.10f}")
    
    # Побудова графіків
    print("\nПобудова графіків...")
    plot_function_and_solution(schaffer_function_n2, best_solution, bounds)
    plot_convergence(ga.history)
    
    return best_solution, best_value


if __name__ == "__main__":
    main()

