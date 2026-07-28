"""

This module contains the Tracker class, which is used to track the progress of optimization algorithms. It provides a
way to store and display the current iteration, log-likelihood, and other relevant information during the optimization
process. The Tracker class can be used in various optimization algorithms, such as logistic regression, to monitor
convergence and performance.

"""

class OptimizationTracker(object):
    """Class to track the cost-function value."""
    def __init__(self):
        self.current_cost = float('-inf')
        self.current_cost_index = 0


