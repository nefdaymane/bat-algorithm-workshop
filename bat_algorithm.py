import numpy as np

class AdvancedRealTimeBatAlgorithm:
    """
    This class implements an advanced real-time Bat Algorithm (BA) for optimization tasks. 
    It leverages techniques like Levy flight initialization, adaptive frequency, 
    loudness, and pulse rate to improve optimization performance.
    """

    def __init__(self, obj_function, n_bats=30, max_iter=150, bounds=((-10, 10), (-10, 10))):
        """
        Initialize the algorithm parameters and population.
        
        Args:
            obj_function (callable): The objective function to minimize.
            n_bats (int): Number of bats in the population.
            max_iter (int): Maximum number of iterations.
            bounds (tuple): Bounds for the search space.
        """
        np.random.seed(42)
        self.obj_function = obj_function
        self.n_bats = n_bats
        self.max_iter = max_iter
        self.bounds = bounds
        self.dimension = 2  # Fixed for this implementation

        # Initialize algorithm state
        self.population = self._levy_flight_initialization()
        self.velocities = np.zeros((n_bats, self.dimension))
        self.fitness = np.array([self.obj_function(x) for x in self.population])
        
        # Track the best solution
        self.best_position = self.population[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)
        
        # History tracking for analysis
        self.iteration_history = []
        self.frequency_history = []
        self.loudness_history = []
        self.pulse_rate_history = []

    def _levy_flight_initialization(self):
        """
        Initialize the population using Levy flight for better diversity.
        
        Returns:
            np.ndarray: Initialized population of bats.
        """
        alpha = 1.5
        sigma_u = np.power(
            np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
            (np.math.gamma((1 + alpha) / 2) * alpha * np.power(2, (alpha - 1) / 2)),
            1 / alpha
        )
        
        population = np.zeros((self.n_bats, self.dimension))
        for i in range(self.n_bats):
            u = np.random.normal(0, sigma_u, self.dimension)
            v = np.random.normal(0, 1, self.dimension)
            step = u / np.power(np.abs(v), 1 / alpha)
            population[i] = np.clip(
                self.bounds[0][0] + np.abs(step) * (self.bounds[0][1] - self.bounds[0][0]),
                self.bounds[0][0],
                self.bounds[0][1]
            )
        return population

    def _update_frequency(self):
        """
        Calculate the adaptive frequency for the bat.
        
        Returns:
            float: The calculated frequency.
        """
        return 0.1 + 0.8 * np.random.random()

    def _update_loudness(self):
        """
        Calculate the loudness of the bat dynamically.
        
        Returns:
            float: The calculated loudness.
        """
        return 0.5 * np.random.random()

    def _update_pulse_rate(self, iteration):
        """
        Calculate the pulse rate of the bat based on iteration.
        
        Args:
            iteration (int): Current iteration index.
        
        Returns:
            float: The calculated pulse rate.
        """
        return 0.5 * (1 - np.exp(-iteration / 10))

    def _update_velocity_and_position(self, i, frequency):
        """
        Update the velocity and position of a bat.
        
        Args:
            i (int): Index of the bat.
            frequency (float): Current frequency for the bat.
        
        Returns:
            np.ndarray: Updated position of the bat.
        """
        self.velocities[i] = (
            0.5 * self.velocities[i] + frequency * (self.population[i] - self.best_position)
        )
        new_position = self.population[i] + self.velocities[i]
        new_position += np.random.uniform(-0.1, 0.1, self.dimension)
        return np.clip(new_position, self.bounds[0][0], self.bounds[0][1])

    def _evaluate_and_update(self, i, new_position, loudness, pulse_rate):
        """
        Evaluate the fitness of a new position and update the population.
        
        Args:
            i (int): Index of the bat.
            new_position (np.ndarray): New position to evaluate.
            loudness (float): Current loudness of the bat.
            pulse_rate (float): Current pulse rate of the bat.
        """
        new_fitness = self.obj_function(new_position)
        if (new_fitness < self.fitness[i] or np.random.random() < loudness * pulse_rate):
            self.population[i] = new_position
            self.fitness[i] = new_fitness
        if new_fitness < self.best_fitness:
            self.best_position = new_position
            self.best_fitness = new_fitness

    def optimize(self):
        """
        Perform the optimization process using the Bat Algorithm.
        
        Returns:
            tuple: Best position and best fitness found.
        """
        for iteration in range(self.max_iter):
            for i in range(self.n_bats):
                frequency = self._update_frequency()
                loudness = self._update_loudness()
                pulse_rate = self._update_pulse_rate(iteration)
                
                new_position = self._update_velocity_and_position(i, frequency)
                self._evaluate_and_update(i, new_position, loudness, pulse_rate)
            
            # Store historical data for visualization
            self.iteration_history.append({
                'population': self.population.copy(),
                'best_position': self.best_position.copy(),
                'best_fitness': self.best_fitness
            })
            self.frequency_history.append(frequency)
            self.loudness_history.append(loudness)
            self.pulse_rate_history.append(pulse_rate)
        
        return self.best_position, self.best_fitness
