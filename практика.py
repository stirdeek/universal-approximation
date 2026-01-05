"""
УНИВЕРСАЛЬНАЯ АППРОКСИМАЦИЯ НЕЙРОННЫМИ СЕТЯМИ
Эксперименты: ReLU vs Sigmoid, Глубина vs Ширина, 2D функции
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import os
import json
from mpl_toolkits.mplot3d import Axes3D

# Настройки отображения графиков
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 10
np.random.seed(42)

# Создаем папку для сохранения графиков
os.makedirs('plots', exist_ok=True)

# 2. ТЕСТОВЫЕ ФУНКЦИИ (1D и 2D)


class TestFunctions:
    """Класс тестовых функций для аппроксимации"""
    
    @staticmethod
    def gaussian_1d(x, mu=0.0, sigma=0.5):
        """Гладкая функция: Гауссова кривая"""
        return np.exp(-((x - mu)**2) / (2 * sigma**2))
    
    @staticmethod
    def trigonometric_1d(x):
        """Гладкая функция: тригонометрическая"""
        return 0.5 * np.sin(2*np.pi*x) + 0.3 * np.cos(4*np.pi*x)
    
    @staticmethod
    def piecewise_smooth_1d(x):
        """Кусочно-гладкая функция с разрывом производной"""
        y = np.zeros_like(x)
        mask = x < 0
        y[mask] = 0.4 * np.sin(3*np.pi*x[mask])
        y[~mask] = 0.6 * np.cos(2*np.pi*x[~mask])
        return y
    
    @staticmethod
    def kinks_1d(x):
        """Функция с локальными изломами"""
        y = 0.3 * np.maximum(0, x + 0.5)
        y += 0.2 * np.maximum(0, x)
        y += 0.4 * np.maximum(0, x - 0.5)
        return y - 0.3
    
    @staticmethod
    def radial_2d(x, y):
        """2D радиальная функция (радиально-симметричная)"""
        r = np.sqrt(x**2 + y**2)
        return np.exp(-r**2)
    
    @staticmethod
    def ridge_2d(x, y):
        """2D ридж-функция (разделимая)"""
        return 0.5 * np.sin(2*x) + 0.3 * np.cos(3*y)
    
    @staticmethod
    def mixed_2d(x, y):
        """2D смешанная функция"""
        return 0.4 * np.exp(-0.5*(x**2 + y**2)) + 0.3 * np.sin(x) * np.cos(y)


# 3. ЭКСПЕРИМЕНТ 1: ReLU vs SIGMOID (1D)


def experiment_relu_vs_sigmoid():
    """Сравнение ReLU и сигмоиды на 1D функциях"""
    
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 1: ReLU vs Sigmoid (1D функции)")
    print("="*70)
    
    n_points = 200
    X = np.linspace(-1, 1, n_points).reshape(-1, 1)
    
    functions = {
        'Гаусс': TestFunctions.gaussian_1d,
        'Тригонометрическая': TestFunctions.trigonometric_1d,
        'Кусочно-гладкая': TestFunctions.piecewise_smooth_1d,
        'С изломами': TestFunctions.kinks_1d
    }
    
    results = {}
    
    for func_name, func in functions.items():
        print(f"\n{func_name}")
        
        y_true = func(X.flatten())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, n_neurons in enumerate([10, 30, 50]):
            ax = axes[idx]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # ReLU сеть
            relu_net = MLPRegressor(
                hidden_layer_sizes=(n_neurons,),
                activation='relu',
                max_iter=2000,
                random_state=42,
                alpha=0.001,
                learning_rate_init=0.01,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20
            )
            
            start_time = time.time()
            relu_net.fit(X_scaled, y_true)
            relu_time = time.time() - start_time
            y_relu = relu_net.predict(X_scaled)
            
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_true.reshape(-1, 1)).flatten()

            sigmoid_net = MLPRegressor(
                hidden_layer_sizes=(n_neurons,),
                activation='logistic',
                max_iter=5000,
                random_state=42,
                alpha=0.001,
                learning_rate_init=0.01,
                solver='lbfgs', 
                validation_fraction=0.3,
                early_stopping=False
            )

            sigmoid_net.fit(X_scaled, y_scaled)  # Обучаем на нормализованных y
            y_sigmoid_scaled = sigmoid_net.predict(X_scaled)
            y_sigmoid = scaler_y.inverse_transform(y_sigmoid_scaled.reshape(-1, 1)).flatten()
            start_time = time.time()
            sigmoid_net.fit(X_scaled, y_true)
            sigmoid_time = time.time() - start_time
            y_sigmoid = sigmoid_net.predict(X_scaled)
            
            # Метрики
            l2_relu = np.sqrt(mean_squared_error(y_true, y_relu))
            linf_relu = np.max(np.abs(y_true - y_relu))
            
            l2_sigmoid = np.sqrt(mean_squared_error(y_true, y_sigmoid))
            linf_sigmoid = np.max(np.abs(y_true - y_sigmoid))
            
            # Сохраняем результаты
            if func_name not in results:
                results[func_name] = []
            
            results[func_name].append({
                'n_neurons': n_neurons,
                'relu': {'l2': l2_relu, 'linf': linf_relu, 'time': relu_time},
                'sigmoid': {'l2': l2_sigmoid, 'linf': linf_sigmoid, 'time': sigmoid_time}
            })
            
            # График с метриками
            ax.plot(X, y_true, 'k-', linewidth=2, label='Истинная', alpha=0.8)
            ax.plot(X, y_relu, 'r--', linewidth=1.5, 
                    label=f'ReLU\nL²={l2_relu:.3f}\nL∞={linf_relu:.3f}')
            ax.plot(X, y_sigmoid, 'b:', linewidth=1.5, 
                    label=f'Sigmoid\nL²={l2_sigmoid:.3f}\nL∞={linf_sigmoid:.3f}')
            
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'{n_neurons} нейронов')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
        
        plt.suptitle(f'Аппроксимация: {func_name} (ReLU vs Sigmoid)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/1d_relu_vs_sigmoid_{func_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  ReLU: L²={results[func_name][-1]['relu']['l2']:.4f}, L∞={results[func_name][-1]['relu']['linf']:.4f}, время={results[func_name][-1]['relu']['time']:.2f}с")
        print(f"  Sigmoid: L²={results[func_name][-1]['sigmoid']['l2']:.4f}, L∞={results[func_name][-1]['sigmoid']['linf']:.4f}, время={results[func_name][-1]['sigmoid']['time']:.2f}с")
    
    # ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: Сводная таблица метрик L∞
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА: L∞ ОШИБКИ")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (func_name, func_data) in enumerate(results.items()):
        ax = axes[idx]
        
        n_neurons = [d['n_neurons'] for d in func_data]
        relu_linf = [d['relu']['linf'] for d in func_data]
        sigmoid_linf = [d['sigmoid']['linf'] for d in func_data]
        
        ax.plot(n_neurons, relu_linf, 'o-', color='red', label='ReLU', linewidth=2, markersize=8)
        ax.plot(n_neurons, sigmoid_linf, 's-', color='blue', label='Sigmoid', linewidth=2, markersize=8)
        
        ax.set_xlabel('Число нейронов')
        ax.set_ylabel('L∞ ошибка')
        ax.set_title(f'{func_name}\nL∞ ошибка vs число нейронов')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Добавляем аннотации
        for i, (r, s) in enumerate(zip(relu_linf, sigmoid_linf)):
            ax.annotate(f'{r:.3f}', (n_neurons[i], r), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8, color='red')
            ax.annotate(f'{s:.3f}', (n_neurons[i], s), textcoords="offset points", 
                       xytext=(0,-15), ha='center', fontsize=8, color='blue')
    
    plt.suptitle('L∞ ошибки для всех тестовых функций', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/linf_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results
# 4. ЭКСПЕРИМЕНТ 2: ЗАВИСИМОСТЬ ОТ ЧИСЛА НЕЙРОНОВ

def experiment_neuron_dependence():
    """Исследование зависимости ошибки от числа нейронов"""
    
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 2: Зависимость ошибки от числа нейронов")
    print("="*70)
    
    # Простая функция для ясной демонстрации
    X = np.linspace(-1, 1, 300).reshape(-1, 1)
    y_true = TestFunctions.gaussian_1d(X.flatten())
    
    neuron_counts = [5, 10, 20, 30, 50, 100]
    
    relu_l2 = []
    sigmoid_l2 = []
    relu_linf = []
    sigmoid_linf = []
    times = []
    
    for n_neurons in neuron_counts:
        print(f"Нейронов: {n_neurons:3d}", end=" | ")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ReLU
        start = time.time()
        relu_net = MLPRegressor(
            hidden_layer_sizes=(n_neurons,),
            activation='relu',
            max_iter=2000,
            random_state=42,
            alpha=0.001
        )
        relu_net.fit(X_scaled, y_true)
        y_relu = relu_net.predict(X_scaled)
        t_relu = time.time() - start
        
        # Sigmoid
        start = time.time()
        sigmoid_net = MLPRegressor(
            hidden_layer_sizes=(n_neurons,),
            activation='logistic',
            max_iter=2000,
            random_state=42,
            alpha=0.001
        )
        sigmoid_net.fit(X_scaled, y_true)
        y_sigmoid = sigmoid_net.predict(X_scaled)
        t_sigmoid = time.time() - start
        
        # Метрики
        l2_r = np.sqrt(mean_squared_error(y_true, y_relu))
        linf_r = np.max(np.abs(y_true - y_relu))
        
        l2_s = np.sqrt(mean_squared_error(y_true, y_sigmoid))
        linf_s = np.max(np.abs(y_true - y_sigmoid))
        
        relu_l2.append(l2_r)
        sigmoid_l2.append(l2_s)
        relu_linf.append(linf_r)
        sigmoid_linf.append(linf_s)
        times.append((t_relu, t_sigmoid))
        
        print(f"  ReLU: L²={l2_r:.4f}, L∞={linf_r:.4f} | Sigmoid: L²={l2_s:.4f}, L∞={linf_s:.4f}")
    
    # Графики
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # L2 ошибка
    ax1 = axes[0, 0]
    ax1.plot(neuron_counts, relu_l2, 'o-', color='red', label='ReLU', linewidth=2)
    ax1.plot(neuron_counts, sigmoid_l2, 's-', color='blue', label='Sigmoid', linewidth=2)
    ax1.set_xlabel('Число нейронов')
    ax1.set_ylabel('L² ошибка (RMSE)')
    ax1.set_title('L² ошибка vs число нейронов', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # L∞ ошибка
    ax2 = axes[0, 1]
    ax2.plot(neuron_counts, relu_linf, 'o-', color='red', label='ReLU', linewidth=2)
    ax2.plot(neuron_counts, sigmoid_linf, 's-', color='blue', label='Sigmoid', linewidth=2)
    ax2.set_xlabel('Число нейронов')
    ax2.set_ylabel('L∞ ошибка (Max Error)')
    ax2.set_title('L∞ ошибка vs число нейронов', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Время обучения
    ax3 = axes[1, 0]
    t_relu = [t[0] for t in times]
    t_sigmoid = [t[1] for t in times]
    ax3.plot(neuron_counts, t_relu, 'o-', color='red', label='ReLU', linewidth=2)
    ax3.plot(neuron_counts, t_sigmoid, 's-', color='blue', label='Sigmoid', linewidth=2)
    ax3.set_xlabel('Число нейронов')
    ax3.set_ylabel('Время обучения (с)')
    ax3.set_title('Время обучения vs число нейронов', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Отношение ошибок
    ax4 = axes[1, 1]
    ratio = [s/r for s, r in zip(sigmoid_l2, relu_l2)]
    ax4.plot(neuron_counts, ratio, 'd-', color='purple', linewidth=2)
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Число нейронов')
    ax4.set_ylabel('Отношение (Sigmoid/ReLU)')
    ax4.set_title('Относительная эффективность', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/neuron_dependence.png', dpi=150, bbox_inches='tight')
    plt.show()
    
     # Таблица результатов
    print("\n" + "="*70)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ: L∞ ОШИБКИ")
    print("="*70)
    print(f"{'Нейроны':>8} | {'ReLU L∞':>12} | {'Sigmoid L∞':>12} | {'Отношение':>10}")
    print("-"*50)
    for n, r_linf, s_linf in zip(neuron_counts, relu_linf, sigmoid_linf):
        ratio = s_linf / r_linf if r_linf > 0 else float('inf')
        print(f"{n:8d} | {r_linf:12.4f} | {s_linf:12.4f} | {ratio:10.2f}")
    
    return {
        'neuron_counts': neuron_counts,
        'relu_l2': relu_l2,
        'sigmoid_l2': sigmoid_l2,
        'relu_linf': relu_linf,      
        'sigmoid_linf': sigmoid_linf, 
        'times': times
    }

# 5. ЭКСПЕРИМЕНТ 3: 2D ФУНКЦИИ

def experiment_2d_functions():
    """Аппроксимация 2D функций"""
    
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 3: 2D функции")
    print("="*70)
    
    # Создаем 2D сетку
    n_points = 20  # Уменьшаем для быстрого обучения
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Подготовка данных для обучения (выравниваем в векторы)
    X_train = np.column_stack([X.flatten(), Y.flatten()])
    
    # 2D тестовые функции
    functions_2d = {
        'Радиальная': TestFunctions.radial_2d(X, Y).flatten(),
        'Ридж': TestFunctions.ridge_2d(X, Y).flatten(),
        'Смешанная': TestFunctions.mixed_2d(X, Y).flatten()
    }
    
    results_2d = {}
    
    for func_name, Z_true in functions_2d.items():
        print(f"\n{func_name} функция:")
        print("-"*30)
        
        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Обучение сети
        model = MLPRegressor(
            hidden_layer_sizes=(50, 30),  # 2 скрытых слоя для 2D
            activation='relu',
            max_iter=2000,
            random_state=42,
            alpha=0.001,
            learning_rate_init=0.01,
            early_stopping=True
        )
        
        start_time = time.time()
        model.fit(X_scaled, Z_true)
        training_time = time.time() - start_time
        
        # Предсказание
        Z_pred = model.predict(X_scaled)
        
        # Метрики
        l2_error = np.sqrt(mean_squared_error(Z_true, Z_pred))
        linf_error = np.max(np.abs(Z_true - Z_pred))
        r2 = r2_score(Z_true, Z_pred)
        
        print(f"  L² ошибка: {l2_error:.4f}")
        print(f"  L∞ ошибка: {linf_error:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Время обучения: {training_time:.2f}с")
        
        # Визуализация
        fig = plt.figure(figsize=(15, 4))
        
        # 1. Истинная функция
        ax1 = fig.add_subplot(131, projection='3d')
        Z_true_reshaped = Z_true.reshape(n_points, n_points)
        surf1 = ax1.plot_surface(X, Y, Z_true_reshaped, cmap='viridis', 
                                alpha=0.8, edgecolor='k', linewidth=0.5)
        ax1.set_title(f'Истинная: {func_name}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        fig.colorbar(surf1, ax=ax1, shrink=0.5)
        
        # 2. Предсказанная функция
        ax2 = fig.add_subplot(132, projection='3d')
        Z_pred_reshaped = Z_pred.reshape(n_points, n_points)
        surf2 = ax2.plot_surface(X, Y, Z_pred_reshaped, cmap='plasma', 
                                alpha=0.8, edgecolor='k', linewidth=0.5)
        ax2.set_title(f'Аппроксимация (L²={l2_error:.3f})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('f̂(x,y)')
        fig.colorbar(surf2, ax=ax2, shrink=0.5)
        
        # 3. Ошибка
        ax3 = fig.add_subplot(133, projection='3d')
        error = np.abs(Z_true_reshaped - Z_pred_reshaped)
        surf3 = ax3.plot_surface(X, Y, error, cmap='hot', 
                                alpha=0.8, edgecolor='k', linewidth=0.5)
        ax3.set_title(f'Абсолютная ошибка (L∞={linf_error:.3f})')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('|f - f̂|')
        fig.colorbar(surf3, ax=ax3, shrink=0.5)
        
        plt.suptitle(f'Аппроксимация 2D функции: {func_name}', 
                    fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(f'plots/2d_{func_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Контурные графики для сравнения
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Истинная функция
        cont1 = axes[0].contourf(X, Y, Z_true_reshaped, 20, cmap='viridis')
        axes[0].set_title('Истинная функция')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        fig.colorbar(cont1, ax=axes[0])
        
        # Предсказанная функция
        cont2 = axes[1].contourf(X, Y, Z_pred_reshaped, 20, cmap='plasma')
        axes[1].set_title('Аппроксимация')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        fig.colorbar(cont2, ax=axes[1])
        
        # Разность
        cont3 = axes[2].contourf(X, Y, error, 20, cmap='RdBu_r')
        axes[2].set_title('Ошибка аппроксимации')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        fig.colorbar(cont3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'plots/2d_contour_{func_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        results_2d[func_name] = {
            'l2': l2_error,
            'linf': linf_error,
            'r2': r2,
            'time': training_time
        }
    
    return results_2d

# 6. ЭКСПЕРИМЕНТ 4: ГЛУБИНА vs ШИРИНА

def experiment_depth_vs_width():
    """Сравнение глубоких и широких сетей"""
    
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 4: Глубина vs Ширина")
    print("="*70)
    
    # Простая тестовая функция
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_true = TestFunctions.kinks_1d(X.flatten())
    
    # Конфигурации сетей с примерно одинаковым числом параметров
    configs = [
        {'name': 'Широкая', 'layers': [100], 'approx_params': 100*2 + 101},
        {'name': '2 слоя', 'layers': [50, 30], 'approx_params': 50*51 + 51*31 + 32},
        {'name': '3 слоя', 'layers': [30, 20, 15], 'approx_params': 30*31 + 31*21 + 21*16 + 17}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}: {config['layers']}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = MLPRegressor(
            hidden_layer_sizes=tuple(config['layers']),
            activation='relu',
            max_iter=2000,
            random_state=42,
            alpha=0.001,
            learning_rate_init=0.01
        )
        
        start_time = time.time()
        model.fit(X_scaled, y_true)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_scaled)
        
        # Метрики
        l2_error = np.sqrt(mean_squared_error(y_true, y_pred))
        linf_error = np.max(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"  L²: {l2_error:.4f}, L∞: {linf_error:.4f}, R²: {r2:.4f}")
        print(f"  Время: {training_time:.2f}с")
        
        results.append({
            'name': config['name'],
            'layers': config['layers'],
            'y_pred': y_pred,
            'l2': l2_error,
            'linf': linf_error,
            'r2': r2,
            'time': training_time
        })
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Графики аппроксимации
    for idx, result in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        
        ax.plot(X, y_true, 'k-', linewidth=2, label='Истинная', alpha=0.8)
        ax.plot(X, result['y_pred'], 'r-', linewidth=1.5, label='Аппроксимация')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f"{result['name']}: {result['layers']}\nL²={result['l2']:.4f}, R²={result['r2']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Сравнение метрик
    ax = axes[1, 1]
    names = [r['name'] for r in results]
    l2_values = [r['l2'] for r in results]
    times = [r['time'] for r in results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, l2_values, width, label='L² ошибка', color='skyblue')
    bars2 = ax.bar(x_pos + width/2, times, width, label='Время (с)', color='lightcoral')
    
    ax.set_xlabel('Архитектура')
    ax.set_ylabel('Значения')
    ax.set_title('Сравнение архитектур')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Глубина vs Ширина: сравнение архитектур', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/depth_vs_width.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

# 7. ГЛАВНАЯ ФУНКЦИЯ

def main():
    """Основная функция для запуска всех экспериментов"""
    
    print("ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ УНИВЕРСАЛЬНОЙ АППРОКСИМАЦИИ")
    
    all_results = {}
    
    try:
        print("\n ЗАПУСК ЭКСПЕРИМЕНТОВ")
        print("-"*50)
        
        # Эксперимент 1: ReLU vs Sigmoid
        print("\n1. ReLU vs Sigmoid (1D функции)...")
        results1 = experiment_relu_vs_sigmoid()
        all_results['relu_vs_sigmoid'] = results1
        
        # Эксперимент 2: Зависимость от числа нейронов
        print("\n2. Зависимость от числа нейронов...")
        results2 = experiment_neuron_dependence()
        all_results['neuron_dependence'] = results2
        
        # Эксперимент 3: 2D функции
        print("\n3. 2D функции...")
        results3 = experiment_2d_functions()
        all_results['2d_functions'] = results3
        
        # Эксперимент 4: Глубина vs Ширина
        print("\n4. Глубина vs Ширина...")
        results4 = experiment_depth_vs_width()
        all_results['depth_vs_width'] = results4
        
    except Exception as e:
        print(f"\n Ошибка: {e}")
        print("Убедитесь, что установлены все библиотеки:")
        print("pip install numpy matplotlib scikit-learn")
        return
    
    # Итоги
    print("ИТОГИ ИССЛЕДОВАНИЯ")
    
    print("\n ВЫПОЛНЕНЫ ЭКСПЕРИМЕНТЫ:")
    print("1. Сравнение ReLU и Sigmoid на 4 типах 1D функций")
    print("2. Зависимость ошибки от числа нейронов (L² и L∞)")
    print("3. Аппроксимация 2D функций (радиальной, ридж, смешанной)")
    print("4. Сравнение глубины и ширины сетей")
    
    print("\n МЕТРИКИ:")
    print("- L² ошибка (RMSE): средняя квадратичная ошибка")
    print("- L∞ ошибка (Max Error): максимальная абсолютная ошибка")
    print("- R²: коэффициент детерминации")
    print("- Время обучения")
    
    # Сохраняем результаты
    with open('plots/experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ!")


if __name__ == "__main__":
    main()