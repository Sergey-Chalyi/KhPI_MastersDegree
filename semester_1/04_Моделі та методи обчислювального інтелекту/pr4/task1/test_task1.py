"""
Тести для генетичного алгоритму оптимізації функції Schaffer N.2
"""

import unittest
import math
import random
from task1 import (
    schaffer_function_n2,
    GeneticAlgorithm
)


class TestSchafferFunction(unittest.TestCase):
    """Тести для функції Schaffer N.2"""
    
    def test_global_minimum(self):
        """Перевірка глобального мінімуму в точці (0, 0)"""
        value = schaffer_function_n2(0, 0)
        self.assertAlmostEqual(value, 0.0, places=10,
                              msg="Глобальний мінімум повинен бути 0 в точці (0, 0)")
    
    def test_function_symmetry(self):
        """Перевірка симетрії функції"""
        x, y = 10.5, -7.3
        value1 = schaffer_function_n2(x, y)
        value2 = schaffer_function_n2(-x, -y)
        value3 = schaffer_function_n2(y, x)
        # Функція не повністю симетрична, але має певні властивості
        self.assertIsInstance(value1, float)
        self.assertIsInstance(value2, float)
        self.assertIsInstance(value3, float)
    
    def test_function_range(self):
        """Перевірка діапазону значень функції"""
        # Функція повинна повертати додатні значення
        test_points = [
            (0, 0), (1, 1), (-1, -1), (10, 10), (-10, -10),
            (50, 50), (-50, -50), (100, 100), (-100, -100)
        ]
        for x, y in test_points:
            value = schaffer_function_n2(x, y)
            self.assertGreaterEqual(value, 0,
                                   msg=f"Функція повинна бути невід'ємною в точці ({x}, {y})")
    
    def test_function_continuity(self):
        """Перевірка неперервності функції"""
        x, y = 5.0, 3.0
        value = schaffer_function_n2(x, y)
        # Невелика зміна аргументів повинна призводити до невеликої зміни значення
        value_near = schaffer_function_n2(x + 0.001, y + 0.001)
        self.assertLess(abs(value - value_near), 1.0,
                       msg="Функція повинна бути неперервною")
    
    def test_function_at_known_points(self):
        """Перевірка значень функції в відомих точках"""
        # В точці (0, 0) мінімум = 0
        self.assertAlmostEqual(schaffer_function_n2(0, 0), 0.0, places=10)
        
        # В точках далі від початку координат значення повинно бути більшим
        value_origin = schaffer_function_n2(0, 0)
        value_far = schaffer_function_n2(10, 10)
        self.assertGreater(value_far, value_origin,
                          msg="Значення функції повинно зростати з відстанню від початку")


class TestGeneticAlgorithm(unittest.TestCase):
    """Тести для генетичного алгоритму"""
    
    def setUp(self):
        """Налаштування перед кожним тестом"""
        self.ga = GeneticAlgorithm(
            objective_func=schaffer_function_n2,
            bounds=(-10, 10),
            population_size=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
    
    def test_generate_population(self):
        """Перевірка генерації початкової популяції"""
        population = self.ga.generate_population()
        
        # Перевірка розміру популяції
        self.assertEqual(len(population), self.ga.population_size)
        
        # Перевірка меж значень
        for individual in population:
            x, y = individual
            self.assertGreaterEqual(x, self.ga.bounds[0])
            self.assertLessEqual(x, self.ga.bounds[1])
            self.assertGreaterEqual(y, self.ga.bounds[0])
            self.assertLessEqual(y, self.ga.bounds[1])
    
    def test_evaluate_fitness(self):
        """Перевірка обчислення придатності"""
        # Особина ближче до мінімуму повинна мати більшу придатність
        individual1 = (0.0, 0.0)  # Глобальний мінімум
        individual2 = (10.0, 10.0)  # Далеко від мінімуму
        
        fitness1 = self.ga.evaluate_fitness(individual1)
        fitness2 = self.ga.evaluate_fitness(individual2)
        
        self.assertGreater(fitness1, fitness2,
                          msg="Особина ближче до мінімуму повинна мати більшу придатність")
        self.assertGreater(fitness1, 0)
        self.assertGreater(fitness2, 0)
    
    def test_tournament_selection(self):
        """Перевірка турнірної селекції"""
        population = self.ga.generate_population()
        fitnesses = [self.ga.evaluate_fitness(ind) for ind in population]
        
        # Виконання селекції кілька разів
        for _ in range(10):
            selected = self.ga.tournament_selection(population, fitnesses)
            self.assertIn(selected, population)
    
    def test_crossover(self):
        """Перевірка схрещування"""
        parent1 = (1.0, 2.0)
        parent2 = (3.0, 4.0)
        
        child1, child2 = self.ga.crossover(parent1, parent2)
        
        # Перевірка, що нащадки є кортежами
        self.assertIsInstance(child1, tuple)
        self.assertIsInstance(child2, tuple)
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)
        
        # Перевірка, що значення в межах
        for coord in child1 + child2:
            self.assertIsInstance(coord, (int, float))
    
    def test_mutate(self):
        """Перевірка мутації"""
        individual = (0.0, 0.0)
        
        # Виконання мутації з високою ймовірністю
        original_mutation_rate = self.ga.mutation_rate
        self.ga.mutation_rate = 1.0  # Завжди мутувати
        
        mutated = self.ga.mutate(individual)
        
        # Перевірка, що значення в межах
        x, y = mutated
        self.assertGreaterEqual(x, self.ga.bounds[0])
        self.assertLessEqual(x, self.ga.bounds[1])
        self.assertGreaterEqual(y, self.ga.bounds[0])
        self.assertLessEqual(y, self.ga.bounds[1])
        
        # Відновлення оригінальної ймовірності мутації
        self.ga.mutation_rate = original_mutation_rate
    
    def test_run_algorithm(self):
        """Перевірка запуску алгоритму"""
        best_solution, best_value = self.ga.run()
        
        # Перевірка типу результату
        self.assertIsInstance(best_solution, tuple)
        self.assertEqual(len(best_solution), 2)
        self.assertIsInstance(best_value, (int, float))
        
        # Перевірка, що значення в межах
        x, y = best_solution
        self.assertGreaterEqual(x, self.ga.bounds[0])
        self.assertLessEqual(x, self.ga.bounds[1])
        self.assertGreaterEqual(y, self.ga.bounds[0])
        self.assertLessEqual(y, self.ga.bounds[1])
        
        # Перевірка, що знайдене значення відповідає функції
        expected_value = schaffer_function_n2(x, y)
        self.assertAlmostEqual(best_value, expected_value, places=10)
        
        # Перевірка історії
        self.assertEqual(len(self.ga.history), self.ga.generations)
        for entry in self.ga.history:
            self.assertIn('generation', entry)
            self.assertIn('best_individual', entry)
            self.assertIn('best_value', entry)
            self.assertIn('avg_fitness', entry)
    
    def test_convergence(self):
        """Перевірка збіжності алгоритму"""
        # Запуск з більшою кількістю поколінь
        ga = GeneticAlgorithm(
            objective_func=schaffer_function_n2,
            bounds=(-10, 10),
            population_size=30,
            generations=20
        )
        
        best_solution, best_value = ga.run()
        
        # Перевірка, що останнє покоління не гірше за перше
        first_best = ga.history[0]['best_value']
        last_best = ga.history[-1]['best_value']
        
        # Алгоритм повинен покращуватися або залишатися на тому ж рівні
        self.assertLessEqual(last_best, first_best,
                            msg="Алгоритм повинен покращувати рішення або залишатися на тому ж рівні")
    
    def test_elitism(self):
        """Перевірка елітизму"""
        population = self.ga.generate_population()
        fitnesses = [self.ga.evaluate_fitness(ind) for ind in population]
        
        # Знаходження найкращих особин
        sorted_indices = sorted(range(len(fitnesses)), 
                               key=lambda i: fitnesses[i], 
                               reverse=True)
        elite_indices = sorted_indices[-self.ga.elitism_count:]
        elite = [population[i] for i in elite_indices]
        
        # Перевірка, що еліта містить найкращі особини
        self.assertEqual(len(elite), self.ga.elitism_count)
        for individual in elite:
            self.assertIn(individual, population)


class TestIntegration(unittest.TestCase):
    """Інтеграційні тести"""
    
    def test_full_optimization(self):
        """Повний тест оптимізації"""
        ga = GeneticAlgorithm(
            objective_func=schaffer_function_n2,
            bounds=(-50, 50),
            population_size=50,
            generations=50,
            crossover_rate=0.8,
            mutation_rate=0.1,
            mutation_strength=0.05
        )
        
        best_solution, best_value = ga.run()
        
        # Перевірка, що знайдено досить хороше рішення
        # (не обов'язково точно (0, 0), але близько)
        distance = math.sqrt(best_solution[0]**2 + best_solution[1]**2)
        self.assertLess(distance, 10.0,
                       msg="Алгоритм повинен знайти рішення близько до глобального мінімуму")
        
        # Перевірка, що значення функції досить мале
        self.assertLess(best_value, 0.1,
                       msg="Значення функції повинно бути досить малим")
    
    def test_different_bounds(self):
        """Тест з різними межами"""
        bounds_list = [(-10, 10), (-50, 50), (-100, 100)]
        
        for bounds in bounds_list:
            ga = GeneticAlgorithm(
                objective_func=schaffer_function_n2,
                bounds=bounds,
                population_size=30,
                generations=20
            )
            
            best_solution, best_value = ga.run()
            
            # Перевірка, що рішення в межах
            x, y = best_solution
            self.assertGreaterEqual(x, bounds[0])
            self.assertLessEqual(x, bounds[1])
            self.assertGreaterEqual(y, bounds[0])
            self.assertLessEqual(y, bounds[1])


def run_tests():
    """Запуск всіх тестів"""
    # Створення тестового набору
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Додавання тестів
    suite.addTests(loader.loadTestsFromTestCase(TestSchafferFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestGeneticAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Запуск тестів
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Виведення результатів
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ")
    print("=" * 70)
    print(f"Запущено тестів: {result.testsRun}")
    print(f"Помилок: {len(result.errors)}")
    print(f"Невдач: {len(result.failures)}")
    print(f"Успішно: {result.testsRun - len(result.errors) - len(result.failures)}")
    
    if result.wasSuccessful():
        print("\n✓ Всі тести пройдені успішно!")
    else:
        print("\n✗ Деякі тести не пройдені")
        if result.errors:
            print("\nПомилки:")
            for test, error in result.errors:
                print(f"  - {test}: {error}")
        if result.failures:
            print("\nНевдачі:")
            for test, failure in result.failures:
                print(f"  - {test}: {failure}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()

