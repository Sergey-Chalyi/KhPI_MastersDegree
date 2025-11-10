"""
Тести для задачі лінійного програмування (кондитерська фабрика)
"""

import unittest
import numpy as np
from scipy.optimize import linprog
from task2 import (
    create_linear_programming_model,
    solve_with_linprog,
    solve_with_differential_evolution,
    compare_solutions,
    objective_function_for_de,
    constraints_for_de
)


class TestLinearProgrammingModel(unittest.TestCase):
    """Тести для математичної моделі"""
    
    def test_model_creation(self):
        """Перевірка створення моделі"""
        model = create_linear_programming_model()
        
        # Перевірка наявності всіх ключів
        self.assertIn('c', model)
        self.assertIn('A_ub', model)
        self.assertIn('b_ub', model)
        self.assertIn('bounds', model)
        
        # Перевірка розмірів
        self.assertEqual(len(model['c']), 2)
        self.assertEqual(model['A_ub'].shape, (3, 2))
        self.assertEqual(len(model['b_ub']), 3)
        self.assertEqual(len(model['bounds']), 2)
    
    def test_objective_function_coefficients(self):
        """Перевірка коефіцієнтів цільової функції"""
        model = create_linear_programming_model()
        # Для мінімізації -Z, коефіцієнти повинні бути від'ємними
        self.assertEqual(model['c'][0], -108)
        self.assertEqual(model['c'][1], -112)
    
    def test_constraints_matrix(self):
        """Перевірка матриці обмежень"""
        model = create_linear_programming_model()
        A_ub = model['A_ub']
        
        # Перевірка коефіцієнтів обмежень
        np.testing.assert_array_almost_equal(A_ub[0], [0.8, 0.5])  # Цукор
        np.testing.assert_array_almost_equal(A_ub[1], [0.4, 0.4])  # Патока
        np.testing.assert_array_almost_equal(A_ub[2], [0.0, 0.1])  # Пюре
    
    def test_constraints_rhs(self):
        """Перевірка правих частин обмежень"""
        model = create_linear_programming_model()
        b_ub = model['b_ub']
        
        self.assertEqual(b_ub[0], 800)  # Цукор
        self.assertEqual(b_ub[1], 600)  # Патока
        self.assertEqual(b_ub[2], 120)  # Пюре
    
    def test_bounds(self):
        """Перевірка меж змінних"""
        model = create_linear_programming_model()
        bounds = model['bounds']
        
        # Обидві змінні повинні бути невід'ємними
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[1][0], 0)
        self.assertIsNone(bounds[0][1])  # Верхня межа не обмежена
        self.assertIsNone(bounds[1][1])


class TestLinprogSolution(unittest.TestCase):
    """Тести для розв'язання методом linprog"""
    
    def test_linprog_solves_successfully(self):
        """Перевірка успішного розв'язання"""
        model = create_linear_programming_model()
        result = solve_with_linprog(model)
        
        self.assertTrue(result['success'])
        self.assertIn('x', result)
        self.assertIn('fun', result)
    
    def test_solution_satisfies_constraints(self):
        """Перевірка, що рішення задовольняє обмеження"""
        model = create_linear_programming_model()
        result = solve_with_linprog(model)
        
        if result['success']:
            x = result['x']
            # Перевірка обмежень
            constraints = model['A_ub'] @ x
            
            for i, constraint_value in enumerate(constraints):
                self.assertLessEqual(constraint_value, model['b_ub'][i] + 1e-6,
                                   msg=f"Обмеження {i} порушено")
    
    def test_solution_non_negative(self):
        """Перевірка невід'ємності змінних"""
        model = create_linear_programming_model()
        result = solve_with_linprog(model)
        
        if result['success']:
            x = result['x']
            self.assertGreaterEqual(x[0], -1e-6, msg="x₁ повинна бути невід'ємною")
            self.assertGreaterEqual(x[1], -1e-6, msg="x₂ повинна бути невід'ємною")
    
    def test_objective_function_value(self):
        """Перевірка значення цільової функції"""
        model = create_linear_programming_model()
        result = solve_with_linprog(model)
        
        if result['success']:
            x = result['x']
            # Обчислюємо значення цільової функції
            z = 108 * x[0] + 112 * x[1]
            # Порівнюємо з результатом (з урахуванням того, що ми максимізуємо)
            self.assertAlmostEqual(z, result['fun'], places=5)


class TestDifferentialEvolutionSolution(unittest.TestCase):
    """Тести для розв'язання методом differential_evolution"""
    
    def test_differential_evolution_solves_successfully(self):
        """Перевірка успішного розв'язання"""
        model = create_linear_programming_model()
        result = solve_with_differential_evolution(model)
        
        self.assertTrue(result['success'])
        self.assertIn('x', result)
        self.assertIn('fun', result)
    
    def test_solution_satisfies_constraints(self):
        """Перевірка, що рішення задовольняє обмеження"""
        model = create_linear_programming_model()
        result = solve_with_differential_evolution(model)
        
        if result['success']:
            x = result['x']
            # Перевірка обмежень
            constraints = model['A_ub'] @ x
            
            for i, constraint_value in enumerate(constraints):
                self.assertLessEqual(constraint_value, model['b_ub'][i] + 1e-3,
                                   msg=f"Обмеження {i} порушено")
    
    def test_solution_non_negative(self):
        """Перевірка невід'ємності змінних"""
        model = create_linear_programming_model()
        result = solve_with_differential_evolution(model)
        
        if result['success']:
            x = result['x']
            self.assertGreaterEqual(x[0], -1e-3, msg="x₁ повинна бути невід'ємною")
            self.assertGreaterEqual(x[1], -1e-3, msg="x₂ повинна бути невід'ємною")


class TestSolutionComparison(unittest.TestCase):
    """Тести для порівняння рішень"""
    
    def test_solutions_are_close(self):
        """Перевірка, що рішення збігаються"""
        model = create_linear_programming_model()
        result_linprog = solve_with_linprog(model)
        result_de = solve_with_differential_evolution(model)
        
        if result_linprog['success'] and result_de['success']:
            are_close = compare_solutions(result_linprog, result_de, tolerance=1e-2)
            # Для differential_evolution може бути трохи менша точність
            self.assertTrue(are_close or 
                          np.allclose(result_linprog['x'], result_de['x'], atol=1e-1))


class TestObjectiveFunction(unittest.TestCase):
    """Тести для цільової функції"""
    
    def test_objective_function_at_origin(self):
        """Перевірка цільової функції в початку координат"""
        x = np.array([0, 0])
        value = objective_function_for_de(x)
        self.assertAlmostEqual(value, 0.0, places=10)
    
    def test_objective_function_at_test_point(self):
        """Перевірка цільової функції в тестовій точці"""
        x = np.array([100, 200])
        value = objective_function_for_de(x)
        expected = -(108 * 100 + 112 * 200)  # Мінус для максимізації
        self.assertAlmostEqual(value, expected, places=10)


class TestConstraints(unittest.TestCase):
    """Тести для обмежень"""
    
    def test_constraints_at_origin(self):
        """Перевірка обмежень в початку координат"""
        x = np.array([0, 0])
        c1, c2, c3 = constraints_for_de(x)
        
        self.assertAlmostEqual(c1, -800, places=10)  # 0 - 800 = -800 ≤ 0 ✓
        self.assertAlmostEqual(c2, -600, places=10)  # 0 - 600 = -600 ≤ 0 ✓
        self.assertAlmostEqual(c3, -120, places=10)  # 0 - 120 = -120 ≤ 0 ✓
    
    def test_constraints_at_feasible_point(self):
        """Перевірка обмежень у допустимій точці"""
        x = np.array([500, 500])
        c1, c2, c3 = constraints_for_de(x)
        
        # Перевірка, що обмеження виконуються (≤ 0)
        self.assertLessEqual(c1, 1e-6)
        self.assertLessEqual(c2, 1e-6)
        self.assertLessEqual(c3, 1e-6)
    
    def test_constraints_at_infeasible_point(self):
        """Перевірка обмежень у недопустимій точці"""
        x = np.array([2000, 2000])
        c1, c2, c3 = constraints_for_de(x)
        
        # Хоча б одне обмеження повинно бути порушено (> 0)
        self.assertTrue(c1 > 0 or c2 > 0 or c3 > 0)


class TestFeasibility(unittest.TestCase):
    """Тести для перевірки допустимості"""
    
    def test_feasible_region_exists(self):
        """Перевірка, що область допустимих рішень не порожня"""
        model = create_linear_programming_model()
        
        # Перевірка, що початок координат допустимий
        x0 = np.array([0, 0])
        constraints = model['A_ub'] @ x0
        self.assertTrue(np.all(constraints <= model['b_ub'] + 1e-10))
    
    def test_optimal_solution_is_feasible(self):
        """Перевірка, що оптимальне рішення допустиме"""
        model = create_linear_programming_model()
        result = solve_with_linprog(model)
        
        if result['success']:
            x = result['x']
            constraints = model['A_ub'] @ x
            
            # Всі обмеження повинні виконуватися
            for i, constraint_value in enumerate(constraints):
                self.assertLessEqual(constraint_value, model['b_ub'][i] + 1e-6,
                                   msg=f"Обмеження {i} порушено в оптимальній точці")


def run_tests():
    """Запуск всіх тестів"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Додавання тестів
    suite.addTests(loader.loadTestsFromTestCase(TestLinearProgrammingModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLinprogSolution))
    suite.addTests(loader.loadTestsFromTestCase(TestDifferentialEvolutionSolution))
    suite.addTests(loader.loadTestsFromTestCase(TestSolutionComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestObjectiveFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestFeasibility))
    
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

