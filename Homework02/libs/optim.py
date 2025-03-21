import numpy as np
from tqdm import tqdm


def fit(model, x : np.array, y : np.array, x_val:np.array = None, y_val:np.array = None, lr: float = 0.5, num_steps : int = 500, show = False):
    """
    Function to fit the logistic regression model using gradient ascent.

    Args:
        model: the logistic regression model.
        x: it's the input data matrix.
        y: the label array.
        x_val: it's the input data matrix for validation.
        y_val: the label array for validation.
        lr: the learning rate.
        num_steps: the number of iterations.

    Returns:
        history: the values of the log likelihood during the process.
    """
    likelihood_history = np.zeros(num_steps)
    val_loss_history = np.zeros(num_steps)

    # Wrap the iterator with tqdm conditionally
    iterator = tqdm(range(num_steps), desc="Training Progress") if show else range(num_steps)

    for it in iterator:

        ##############################
        ###     START CODE HERE    ###
        ##############################
        # Retrieve the predicitions
        preds = model.predict(x)
        # Compute and save the likelihood for the current iteration
        likelihood_history[it] = model.likelihood(preds, y)

        # Compute the gradients of the log likelihood
        gradient = model.compute_gradient(x, y, preds)
        # Update the parameters
        model.update_theta(gradient, lr)
        ##############################
        ###      END CODE HERE     ###
        ##############################
        if x_val is not None and y_val is not None:
            val_preds = model.predict(x_val)
            val_loss_history[it] = - model.likelihood(val_preds, y_val)
            
    return likelihood_history, val_loss_history

