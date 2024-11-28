import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from objective_functions import sphere_function

def visualize_optimization(bat_algo):
    plt.figure(figsize=(16, 8), facecolor='#f0f0f0')
    plt.suptitle('Advanced Real-Time Bat Algorithm', fontsize=16)
    gs = plt.GridSpec(2, 2, width_ratios=[2, 1])
    
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = sphere_function([X, Y])
    
    ax1 = plt.subplot(gs[:, 0])
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', norm=LogNorm(), alpha=0.7)
    plt.colorbar(contour, ax=ax1, label='Objective Function Value')
    ax1.set_title('Optimization Landscape', fontsize=12)
    
    population_scatter = ax1.scatter([], [], c='red', s=50, alpha=0.7)
    best_point = ax1.scatter([], [], c='white', s=200, edgecolors='black')
    
    ax2 = plt.subplot(gs[0, 1])
    fitness_line, = ax2.plot([], [], c='blue', linewidth=2)
    ax2.set_title('Best Fitness Progression', fontsize=12)
    ax2.set_yscale('log')
    
    ax3 = plt.subplot(gs[1, 1])
    frequency_line, = ax3.plot([], [], c='green', linewidth=2, label='Frequency')
    loudness_line, = ax3.plot([], [], c='orange', linewidth=2, label='Loudness')
    pulse_rate_line, = ax3.plot([], [], c='purple', linewidth=2, label='Pulse Rate')
    ax3.set_title('Algorithm Parameters', fontsize=12)
    ax3.legend()

    def update(frame):
        iteration_data = bat_algo.iteration_history[frame]
        population_scatter.set_offsets(iteration_data['population'])
        best_point.set_offsets(iteration_data['best_position'])
        
        fitness_history = [step['best_fitness'] for step in bat_algo.iteration_history[:frame+1]]
        fitness_line.set_data(range(len(fitness_history)), fitness_history)
        ax2.relim()
        ax2.autoscale_view()
        
        frequency_line.set_data(range(frame + 1), bat_algo.frequency_history[:frame + 1])
        loudness_line.set_data(range(frame + 1), bat_algo.loudness_history[:frame + 1])
        pulse_rate_line.set_data(range(frame + 1), bat_algo.pulse_rate_history[:frame + 1])
        ax3.relim()
        ax3.autoscale_view()
        
        return population_scatter, best_point, fitness_line, frequency_line, loudness_line, pulse_rate_line

    anim = FuncAnimation(plt.gcf(), update, frames=len(bat_algo.iteration_history), interval=200, repeat=False)
    plt.tight_layout()
    plt.show()
