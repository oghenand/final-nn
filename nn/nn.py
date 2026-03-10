# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # apply weights and add biases (dot product)
        Z = np.dot(A_prev, W_curr.T) + b_curr.T
        # apply activation function
        if activation == 'sigmoid':
            A = self._sigmoid(Z)
        elif activation == 'relu':
            A = self._relu(Z)
        else:
            raise ValueError('Unsupported Activation function! choose between relu and sigmoid')

        return A, Z

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        curr = X
        cache = dict()
        cache['A0'] = X
        cache['Z0'] = X
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']
            activation = layer['activation']

            curr, Z = self._single_forward(W_curr, b_curr, curr, activation)
            cache[f'A{layer_idx}'] = curr
            cache[f'Z{layer_idx}'] = Z
        return curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise ValueError('Unsupported activation function given.')

        dW_curr = np.dot(dZ_curr.T, A_prev)/self._batch_size
        dA_prev = np.dot(dZ_curr, W_curr)
        db_curr = np.sum(dZ_curr, axis=0, keepdims=True).T/self._batch_size

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        max_layer = len(self.arch)
        grad_dict = {}
        if self._loss_func == 'binary_cross_entropy':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError('Unsupported loss function provided!')
        
        for layer_idx in reversed(list(range(1, max_layer+1))):
            A_prev = cache[f'A{layer_idx-1}']
            Z_curr = cache[f'Z{layer_idx}']
            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']
            activation_curr = self.arch[layer_idx-1]['activation']
            
            grads = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr,
                                            activation_curr)
            dA_prev, dW_curr, db_curr = grads
            grad_dict[f'W{layer_idx}'] = dW_curr
            grad_dict[f'b{layer_idx}'] = db_curr

            dA_curr = dA_prev # backprop step
        
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for key in self._param_dict:
            self._param_dict[key] -= self._lr * grad_dict[key]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        loss_func_dict = {'binary_cross_entropy': self._binary_cross_entropy,
                          'mean_squared_error': self._mean_squared_error}
        if self._loss_func not in ('binary_cross_entropy', 'mean_squared_error'):
            raise ValueError('Loss function not supported')
        n_batches = len(X_train) // self._batch_size
        for epoch in range(self._epochs):
            epoch_loss = 0
            curr_idx = 0

            # shuffle X_train and y_train
            idxs = np.arange(len(X_train))
            np.random.shuffle(idxs)
            X_train = X_train[idxs]
            y_train = y_train[idxs]

            while True:
                # TODO: might have to shuffle idxs!
                start_idx = int(curr_idx*self._batch_size)
                end_idx = min(start_idx + self._batch_size, len(X_train))

                if start_idx >= len(X_train):
                    break

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # run forward
                batch_output, batch_cache = self.forward(X_batch)
                # calculate loss
                epoch_loss += loss_func_dict[self._loss_func](y_batch, batch_output)
                # backprop and loss
                grad_dict = self.backprop(y_batch, batch_output, batch_cache)
                # update params
                self._update_params(grad_dict)
                curr_idx +=1

            per_epoch_loss_train.append(epoch_loss)

            # val_loss
            val_output = self.predict(X_val)
            per_epoch_loss_val.append(loss_func_dict[self._loss_func](y_val, val_output))

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sig_Z = self._sigmoid(Z)
        dZ = dA*sig_Z*(1-sig_Z)
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0,Z)
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        d_relu = np.where(Z>0, 1, 0)
        dZ = dA*d_relu
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        if not len(y) > 0 or not len(y_hat) > 0:
            raise ValueError('y and y_hat must not be empty!')
        if not len(y) == len(y_hat):
            raise ValueError('y and y_hat must be the same length!')
        if not ((y==0)|(y==1)).all():
            raise ValueError('y must be binary (0s and 1s!)')
        if not (np.min(y_hat) >= 0 and np.max(y_hat) <= 1):
            raise ValueError('y_hat must be probabilities between 0 and 1!')
        
        # enforce numerical stability w/ np.clip
        y_hat = np.clip(y_hat, 1e-6, 1 - 1e-6)
        loss = -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return loss
    
    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        if not len(y) > 0 or not len(y_hat) > 0:
            raise ValueError('y and y_hat must not be empty!')
        if not len(y) == len(y_hat):
            raise ValueError('y and y_hat must be the same length!')
        if not ((y==0)|(y==1)).all():
            raise ValueError('y must be binary (0s and 1s!)')
        if not (np.min(y_hat) >= 0 and np.max(y_hat) <= 1):
            raise ValueError('y_hat must be probabilities between 0 and 1!')
        
        # enforce numerical stability w/ np.clip
        y_hat = np.clip(y_hat, 1e-6, 1 - 1e-6)
        dA = -y/y_hat + (1-y)/(1-y_hat)
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        if not len(y) > 0 or not len(y_hat) > 0:
            raise ValueError('y and y_hat must not be empty!')
        if not len(y) == len(y_hat):
            raise ValueError('y and y_hat must be the same length!')
        
        loss = np.mean((y-y_hat)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        if not len(y) > 0 or not len(y_hat) > 0:
            raise ValueError('y and y_hat must not be empty!')
        if not len(y) == len(y_hat):
            raise ValueError('y and y_hat must be the same length!')
        dA = -2*(y-y_hat) / len(y)
        return dA