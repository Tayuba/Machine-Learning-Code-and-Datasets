import numpy as np
import matplotlib.pyplot as plt

"""It better to understand what happens internally when using scikit, with this in mind i will calculte the gradient 
descent using function"""

# Assuming the X and Y vectors are given and we want to find the best fit line
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Function to calculate the gradient descent
def gradient_descent(x, y):
    """Consider the MSE Partial Differentiation Formulas
    d/dm(md) = -2/n.sum(x(y-(mx +b)) and d/db(bd) = -2/n.sum(y-(mx +b))
    and the respective formula for m and b,  as m_curr = m_curr - learning rate * d/dm and
    b_curr = b_curr - learning rate * d/db and finally y_predictive = m_curr * x + b_curr"""

    # Initialized the m_curr and b_curr to be zero
    m_curr = b_curr = 0
    # Define number of iterations, this value may be fine tune depending on the outcome
    iteration = 1000
    # length of the data
    n = len(x)
    # Learning rate with some initial value
    learning_rate = 0.08
    for i in range(iteration):
        y_predictive = m_curr * x + b_curr
        # Using cos to check if the gradient descent
        cost_function = (1/n) * sum([var**2 for var in (y - y_predictive)])
        md = -(2/n) * sum(x * (y - y_predictive))
        bd = -(2/n) * sum(y - y_predictive)
        # Adjust m_curr and b_curr
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("M_curr: {}, B_curr: {}, Cost_Function: {} Iteration: {}".format(m_curr, b_curr, cost_function, i))


gradient_descent(x, y)
