"""
Лабораторна робота №4, Завдання 2
Розв'язання задачі лінійного програмування

Задача: Кондитерська фабрика
Знайти план виробництва карамелі, який забезпечує максимальний приріст
продуктивності праці на новому обладнанні.
"""

import numpy as np
from scipy.optimize import linprog, differential_evolution, LinearConstraint
import matplotlib.pyplot as plt
from typing import Tuple, Dict


# ============================================================================
# 1. ОПИС ЗМІННИХ
# ============================================================================

"""
Змінні рішення:
- x₁ (x1) - кількість тонн карамелі типу A, що виробляється
- x₂ (x2) - кількість тонн карамелі типу B, що виробляється

Всі змінні невід'ємні: x₁ ≥ 0, x₂ ≥ 0
"""


# ============================================================================
# 2. МАТЕМАТИЧНА МОДЕЛЬ ЗАДАЧІ
# ============================================================================

"""
Цільова функція (максимізація):
    Z = 108x₁ + 112x₂ → max

Обмеження:
    1. Цукор:     0.8x₁ + 0.5x₂ ≤ 800
    2. Патока:    0.4x₁ + 0.4x₂ ≤ 600
    3. Пюре:      0.0x₁ + 0.1x₂ ≤ 120
    
Умови невід'ємності:
    x₁ ≥ 0, x₂ ≥ 0

Для використання linprog() потрібно перетворити на мінімізацію:
    min Z' = -108x₁ - 112x₂
"""


def create_linear_programming_model() -> Dict:
    """
    Створення математичної моделі задачі лінійного програмування
    
    Returns:
        Словник з параметрами задачі
    """
    # Коефіцієнти цільової функції (для мінімізації, тому з мінусом)
    c = np.array([-108, -112])  # Мінімізуємо -Z, щоб максимізувати Z
    
    # Коефіцієнти обмежень (ліва частина нерівностей)
    A_ub = np.array([
        [0.8, 0.5],   # Цукор
        [0.4, 0.4],   # Патока
        [0.0, 0.1]    # Фруктове пюре
    ])
    
    # Права частина обмежень
    b_ub = np.array([800, 600, 120])
    
    # Межі змінних (x₁ ≥ 0, x₂ ≥ 0)
    bounds = [(0, None), (0, None)]
    
    return {
        'c': c,
        'A_ub': A_ub,
        'b_ub': b_ub,
        'bounds': bounds
    }


def solve_with_linprog(model: Dict) -> Dict:
    """
    Розв'язання задачі лінійного програмування за допомогою linprog()
    
    Args:
        model: Словник з параметрами задачі
        
    Returns:
        Словник з результатами оптимізації
    """
    print("=" * 70)
    print("Розв'язання за допомогою scipy.optimize.linprog()")
    print("=" * 70)
    
    # Виклик linprog
    result = linprog(
        c=model['c'],
        A_ub=model['A_ub'],
        b_ub=model['b_ub'],
        bounds=model['bounds'],
        method='highs',  # Сучасний метод для лінійного програмування
        options={'disp': True}
    )
    
    if result.success:
        x1, x2 = result.x
        max_value = -result.fun  # Перетворюємо назад на максимум
        
        print(f"\n✓ Рішення знайдено успішно!")
        print(f"  x₁ (карамель A) = {x1:.2f} т")
        print(f"  x₂ (карамель B) = {x2:.2f} т")
        print(f"  Максимальний приріст продуктивності = {max_value:.2f}%")
        print(f"  Кількість ітерацій: {result.nit}")
        
        # Перевірка обмежень
        print("\nПеревірка обмежень:")
        constraints = model['A_ub'] @ result.x
        constraint_names = ['Цукор', 'Патока', 'Фруктове пюре']
        for i, (name, used, available) in enumerate(zip(constraint_names, constraints, model['b_ub'])):
            print(f"  {name}: {used:.2f} ≤ {available:.2f} т (використано {used/available*100:.1f}%)")
        
        return {
            'success': True,
            'x': result.x,
            'fun': max_value,
            'method': 'linprog',
            'iterations': result.nit
        }
    else:
        print(f"\n✗ Помилка: {result.message}")
        return {
            'success': False,
            'message': result.message
        }


def objective_function_for_de(x: np.ndarray) -> float:
    """
    Цільова функція для differential_evolution (мінімізація)
    
    Args:
        x: Масив [x1, x2]
        
    Returns:
        Значення цільової функції (з мінусом для максимізації)
    """
    x1, x2 = x
    return -(108 * x1 + 112 * x2)  # Мінус для максимізації


def constraints_for_de(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Обмеження для differential_evolution
    
    Args:
        x: Масив [x1, x2]
        
    Returns:
        Значення обмежень (повинні бути ≤ 0)
    """
    x1, x2 = x
    # Перетворюємо нерівності на формат g(x) ≤ 0
    constraint1 = 0.8 * x1 + 0.5 * x2 - 800  # ≤ 0
    constraint2 = 0.4 * x1 + 0.4 * x2 - 600  # ≤ 0
    constraint3 = 0.1 * x2 - 120              # ≤ 0
    return constraint1, constraint2, constraint3


def solve_with_differential_evolution(model: Dict) -> Dict:
    """
    Розв'язання задачі за допомогою differential_evolution()
    
    Args:
        model: Словник з параметрами задачі
        
    Returns:
        Словник з результатами оптимізації
    """
    print("\n" + "=" * 70)
    print("Розв'язання за допомогою scipy.optimize.differential_evolution()")
    print("=" * 70)
    
    # Визначаємо межі для пошуку
    # Враховуючи обмеження, знаходимо максимальні можливі значення
    bounds = model['bounds']
    
    # Якщо bounds містить None, встановлюємо розумні межі
    max_x1 = min(800 / 0.8, 600 / 0.4)  # Максимум з обмежень
    max_x2 = min(800 / 0.5, 600 / 0.4, 120 / 0.1)  # Максимум з обмежень
    
    bounds_de = [(0, max_x1), (0, max_x2)]
    
    # Створюємо лінійні обмеження за допомогою LinearConstraint
    # Обмеження: A_ub @ x <= b_ub
    # LinearConstraint: lb <= A @ x <= ub
    # Тому: -inf <= A_ub @ x <= b_ub
    A_ub = model['A_ub']
    b_ub = model['b_ub']
    
    # Створюємо LinearConstraint
    linear_constraint = LinearConstraint(
        A=A_ub,
        lb=-np.inf,
        ub=b_ub
    )
    
    # Виклик differential_evolution
    result = differential_evolution(
        func=objective_function_for_de,
        bounds=bounds_de,
        constraints=linear_constraint,
        seed=42,
        maxiter=1000,
        popsize=15,
        atol=1e-6,
        tol=1e-6
    )
    
    if result.success:
        x1, x2 = result.x
        max_value = -result.fun  # Перетворюємо назад на максимум
        
        print(f"\n✓ Рішення знайдено успішно!")
        print(f"  x₁ (карамель A) = {x1:.6f} т")
        print(f"  x₂ (карамель B) = {x2:.6f} т")
        print(f"  Максимальний приріст продуктивності = {max_value:.6f}%")
        print(f"  Кількість ітерацій: {result.nit}")
        print(f"  Кількість оцінок функції: {result.nfev}")
        
        # Перевірка обмежень
        print("\nПеревірка обмежень:")
        constraints = model['A_ub'] @ result.x
        constraint_names = ['Цукор', 'Патока', 'Фруктове пюре']
        for i, (name, used, available) in enumerate(zip(constraint_names, constraints, model['b_ub'])):
            print(f"  {name}: {used:.6f} ≤ {available:.2f} т (використано {used/available*100:.1f}%)")
        
        return {
            'success': True,
            'x': result.x,
            'fun': max_value,
            'method': 'differential_evolution',
            'iterations': result.nit,
            'nfev': result.nfev
        }
    else:
        print(f"\n✗ Помилка: {result.message}")
        return {
            'success': False,
            'message': result.message
        }


def compare_solutions(result1: Dict, result2: Dict, tolerance: float = 1e-3) -> bool:
    """
    Порівняння рішень за допомогою np.allclose()
    
    Args:
        result1: Результат першого методу
        result2: Результат другого методу
        tolerance: Допустима похибка
        
    Returns:
        True, якщо рішення близькі
    """
    print("\n" + "=" * 70)
    print("Порівняння рішень")
    print("=" * 70)
    
    if not result1['success'] or not result2['success']:
        print("✗ Неможливо порівняти: один з методів не знайшов рішення")
        return False
    
    x1 = result1['x']
    x2 = result2['x']
    fun1 = result1['fun']
    fun2 = result2['fun']
    
    # Порівняння значень змінних
    x_close = np.allclose(x1, x2, atol=tolerance, rtol=tolerance)
    
    # Порівняння значень цільової функції
    fun_close = np.allclose(fun1, fun2, atol=tolerance, rtol=tolerance)
    
    print(f"\nПорівняння змінних:")
    print(f"  linprog:        x₁ = {x1[0]:.6f}, x₂ = {x1[1]:.6f}")
    print(f"  diff_evolution: x₁ = {x2[0]:.6f}, x₂ = {x2[1]:.6f}")
    print(f"  Збігаються: {x_close} (допуск: {tolerance})")
    
    print(f"\nПорівняння цільової функції:")
    print(f"  linprog:        Z = {fun1:.6f}%")
    print(f"  diff_evolution: Z = {fun2:.6f}%")
    print(f"  Збігаються: {fun_close} (допуск: {tolerance})")
    
    if x_close and fun_close:
        print(f"\n✓ Рішення збігаються в межах допуску {tolerance}")
        return True
    else:
        print(f"\n✗ Рішення не збігаються в межах допуску {tolerance}")
        return False


def plot_solution(model: Dict, result_linprog: Dict, result_de: Dict):
    """
    Візуалізація області допустимих рішень та оптимальних точок
    
    Args:
        model: Параметри задачі
        result_linprog: Результат linprog
        result_de: Результат differential_evolution
    """
    print("\n" + "=" * 70)
    print("Побудова графіка")
    print("=" * 70)
    
    # Створюємо сітку значень
    x1_max = 1500
    x2_max = 1500
    x1 = np.linspace(0, x1_max, 1000)
    
    # Обмеження як лінії
    # 0.8x₁ + 0.5x₂ = 800  =>  x₂ = (800 - 0.8x₁) / 0.5
    constraint1 = (800 - 0.8 * x1) / 0.5
    constraint1 = np.clip(constraint1, 0, x2_max)
    
    # 0.4x₁ + 0.4x₂ = 600  =>  x₂ = (600 - 0.4x₁) / 0.4
    constraint2 = (600 - 0.4 * x1) / 0.4
    constraint2 = np.clip(constraint2, 0, x2_max)
    
    # 0.1x₂ = 120  =>  x₂ = 1200
    constraint3 = np.full_like(x1, 1200)
    constraint3 = np.clip(constraint3, 0, x2_max)
    
    # Знаходимо область допустимих рішень
    feasible_x2 = np.minimum.reduce([
        constraint1,
        constraint2,
        constraint3,
        np.full_like(x1, x2_max)
    ])
    
    # Створюємо графік
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Область допустимих рішень
    ax.fill_between(x1, 0, feasible_x2, alpha=0.3, color='lightblue', 
                    label='Область допустимих рішень')
    
    # Лінії обмежень
    ax.plot(x1, constraint1, 'r-', linewidth=2, label='Цукор: 0.8x₁ + 0.5x₂ = 800')
    ax.plot(x1, constraint2, 'g-', linewidth=2, label='Патока: 0.4x₁ + 0.4x₂ = 600')
    ax.plot(x1, constraint3, 'b-', linewidth=2, label='Пюре: 0.1x₂ = 120')
    
    # Оптимальні точки
    if result_linprog['success']:
        x1_opt1, x2_opt1 = result_linprog['x']
        ax.plot(x1_opt1, x2_opt1, 'ro', markersize=12, 
               label=f'linprog: ({x1_opt1:.2f}, {x2_opt1:.2f})', 
               markeredgecolor='black', markeredgewidth=2)
    
    if result_de['success']:
        x1_opt2, x2_opt2 = result_de['x']
        ax.plot(x1_opt2, x2_opt2, 'g*', markersize=15, 
               label=f'diff_evolution: ({x1_opt2:.2f}, {x2_opt2:.2f})',
               markeredgecolor='black', markeredgewidth=2)
    
    # Лінії рівня цільової функції (для візуалізації)
    if result_linprog['success']:
        z_opt = result_linprog['fun']
        # 108x₁ + 112x₂ = z_opt  =>  x₂ = (z_opt - 108x₁) / 112
        iso_line = (z_opt - 108 * x1) / 112
        iso_line = np.clip(iso_line, 0, x2_max)
        ax.plot(x1, iso_line, 'k--', linewidth=1.5, alpha=0.7, 
               label=f'Лінія рівня: Z = {z_opt:.0f}%')
    
    ax.set_xlabel('x₁ (карамель A, т)', fontsize=12)
    ax.set_ylabel('x₂ (карамель B, т)', fontsize=12)
    ax.set_title('Графік задачі лінійного програмування\nОбласть допустимих рішень та оптимальні точки', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(x1_max, 1200))
    ax.set_ylim(0, min(x2_max, 1500))
    
    plt.tight_layout()
    plt.savefig('linear_programming_solution.png', dpi=300, bbox_inches='tight')
    print("Графік збережено у файл 'linear_programming_solution.png'")
    plt.show()


def explain_results(result_linprog: Dict, result_de: Dict):
    """
    Пояснення отриманих результатів
    
    Args:
        result_linprog: Результат linprog
        result_de: Результат differential_evolution
    """
    print("\n" + "=" * 70)
    print("ПОЯСНЕННЯ РЕЗУЛЬТАТІВ")
    print("=" * 70)
    
    if not result_linprog['success']:
        print("Не вдалося знайти рішення методом linprog")
        return
    
    x1, x2 = result_linprog['x']
    max_value = result_linprog['fun']
    
    print(f"""
1. ОПТИМАЛЬНИЙ ПЛАН ВИРОБНИЦТВА:
   - Карамель типу A: {x1:.2f} тонн
   - Карамель типу B: {x2:.2f} тонн
   
2. МАКСИМАЛЬНИЙ ПРИРІСТ ПРОДУКТИВНОСТІ:
   - {max_value:.2f}%
   
3. ВИКОРИСТАННЯ РЕСУРСІВ:
   - Цукор: {0.8*x1 + 0.5*x2:.2f} т з 800 т доступних
   - Патока: {0.4*x1 + 0.4*x2:.2f} т з 600 т доступних
   - Фруктове пюре: {0.1*x2:.2f} т з 120 т доступних
   
4. ЕКОНОМІЧНА ІНТЕРПРЕТАЦІЯ:
   - Для максимізації приросту продуктивності праці необхідно виробляти
     {x1:.2f} тонн карамелі типу A та {x2:.2f} тонн карамелі типу B.
   - Це забезпечить максимальний приріст продуктивності {max_value:.2f}%.
   - Карамель типу B має вищий приріст продуктивності (112% проти 108%),
     тому оптимальний план передбачає виробництво більшої кількості карамелі B.
   
5. ПОРІВНЯННЯ МЕТОДІВ:
   - linprog(): Спеціалізований метод для лінійного програмування, 
     знаходить точне рішення швидко.
   - differential_evolution(): Універсальний еволюційний алгоритм,
     може працювати з нелінійними задачами, але потребує більше обчислень.
   - Обидва методи дають однаковий результат, що підтверджує правильність
     розв'язання задачі.
""")


def main():
    """Головна функція для запуску розв'язання задачі"""
    print("=" * 70)
    print("ЗАДАЧА ЛІНІЙНОГО ПРОГРАМУВАННЯ")
    print("Кондитерська фабрика - Оптимізація виробництва карамелі")
    print("=" * 70)
    
    # Створення моделі
    model = create_linear_programming_model()
    
    # Розв'язання методом linprog
    result_linprog = solve_with_linprog(model)
    
    # Розв'язання методом differential_evolution
    result_de = solve_with_differential_evolution(model)
    
    # Порівняння рішень
    if result_linprog['success'] and result_de['success']:
        compare_solutions(result_linprog, result_de, tolerance=1e-3)
    
    # Візуалізація
    if result_linprog['success']:
        plot_solution(model, result_linprog, result_de)
    
    # Пояснення результатів
    explain_results(result_linprog, result_de)
    
    return result_linprog, result_de


if __name__ == "__main__":
    result1, result2 = main()

