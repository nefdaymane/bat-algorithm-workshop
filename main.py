from bat_algorithm import AdvancedRealTimeBatAlgorithm
from objective_functions import sphere_function
from visualization import visualize_optimization

# Execute optimization
bat_algo = AdvancedRealTimeBatAlgorithm(
    obj_function=sphere_function,
    n_bats=50,
    max_iter=200,
    bounds=((-10, 10), (-10, 10))
)
best_pos, best_fit = bat_algo.optimize()

print(f"Best Position: {best_pos}")
print(f"Best Fitness: {best_fit}")

visualize_optimization(bat_algo)
