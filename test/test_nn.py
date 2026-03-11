# TODO: import dependencies and write unit tests below
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
import pytest
# MY IMPORTS
import random

from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

# def params for use for testing
nn_arch = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, 
           {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
lr = 1e-4
seed = 42
batch_size = 32
epochs = 10
loss_function = 'mean_squared_error'

# create simple network for testing
nn_obj = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

def test_single_forward():
    """
    Test cases for single-layer forward pass.
    """
    # create example matrices
    W_curr = np.random.randn(4,8)
    b_curr = np.random.randn(4,1)
    A_prev = np.random.randn(32,8)
    manual_Z = np.dot(A_prev, W_curr.T) + b_curr.T
    manual_A = 1 / (1+np.exp(-manual_Z))
    manual_A_relu = np.maximum(0, manual_Z)

    A_curr, Z_curr = nn_obj._single_forward(W_curr, b_curr, A_prev, 'sigmoid')
    A_curr_relu, _ = nn_obj._single_forward(W_curr, b_curr, A_prev, 'relu')
    assert A_curr.shape == (32,4) and Z_curr.shape == (32,4), "Forward returns unexpected shape!"
    assert A_curr.shape == Z_curr.shape, "Activations and linear transform mats must be same shape!"
    # check that sigmoid forward pass returns values in range of 0 and 1
    assert A_curr.max() <= 1 and A_curr.min() >=0, "Sigmoid-activations have incorrect range!"
    assert A_curr_relu.min() >=0, "Relu activations have incorrect range!"
    # check for closeness to manual results
    assert np.allclose(Z_curr, manual_Z), "Manually-calcualted and NN returned Z arrs mismatched!"
    assert np.allclose(manual_A, A_curr, atol=1e-6), "Manually-calcualted and NN returned A arrs (sigmoid) mismatched!"
    assert np.allclose(manual_A_relu, A_curr_relu, atol=1e-6), "Manually-calcualted and NN returned A arrs (relu) mismatched!"
    # check for unsupported activation function
    with pytest.raises(ValueError):
        nn_obj._single_forward(W_curr, b_curr, A_prev, 'None')
    assert not np.any(np.isnan(A_curr)), "Layer output must not have NaNs"
    assert not np.any(np.isnan(Z_curr)), "Layer linear transform must not have NaNs"

def test_forward():
    """
    Test cases for forward (from input to output).
    """
    X = np.random.randn(nn_obj._batch_size, 64)
    A_curr, cache = nn_obj.forward(X)
    assert A_curr.shape == (X.shape[0], nn_arch[-1]['output_dim']), "NN output shape doesn't match defined architecture."
    assert len(cache) == 2*(len(nn_arch)+1), "NN cache must be 1 + number of defined layers"
    cond = (0,1) if nn_arch[-1]['activation'] == 'sigmoid' else (0,float('inf'))
    assert A_curr.min() >= cond[0] and A_curr.max() <= cond[1], "Output ranges don't match readout activation function"
    assert not np.any(np.isnan(A_curr)), "Output must not have NaNs"
    for v in cache.values():
        assert not np.any(np.isnan(v)), "Cache must not contain NaNs in any array!"

def test_single_backprop():
    """
    Test cases for backpropagation.
    """
    # create example matrices
    W_curr = np.random.randn(4,8)
    b_curr = np.random.randn(4,1)
    Z_curr = np.random.randn(32,4)
    A_prev = np.random.randn(32,8)
    dA_curr = np.random.randn(32,4)

    # do single-layer backprop
    dA_prev, dW_curr, db_curr = nn_obj._single_backprop(W_curr, b_curr, Z_curr,
                                                        A_prev, dA_curr, 'sigmoid')
    
    # check that shapes match (gradients calculated must be same shape as arrays)
    assert dA_prev.shape == A_prev.shape
    assert dW_curr.shape == W_curr.shape
    assert db_curr.shape == b_curr.shape

    # check for NaNs
    arrs = (dA_prev, dW_curr, db_curr)
    # line will return 0 (false) if NaNs, 1 otherwise
    # summing bool array will treat False as 0 and True as 1
    # sum should be equal to number of gradients (3) since no grads should have NaNs
    assert np.sum([not np.any(np.isnan(arr)) for arr in arrs]) == 3, "No gradient should have NaNs!"

def test_predict():
    """
    Test cases for predict function.
    """
    X = np.random.randn(nn_obj._batch_size, 64)
    n_samples = X.shape[0]
    y_hat = nn_obj.predict(X)

    # tests for predict
    assert n_samples == len(y_hat), "Number of predictions must be equal to number of samples!"
    assert not np.any(np.isnan(y_hat)), "Model predicted NaNs; computation is likely wrong!"

def test_binary_cross_entropy():
    """
    Test cases for BCE.
    """
    # use example y_hat and y
    y = np.array([0, 1, 0, 0])
    y_hat = np.array([0.05, 0.96, 0.34, 0.87])
    
    bce_loss = nn_obj._binary_cross_entropy(y, y_hat)
    # clip for manual calc
    y_hat_clip = np.clip(y_hat, 1e-6, 1 - 1e-6) 
    manual_bce = -np.mean(y * np.log(y_hat_clip) + (1 - y) * np.log(1 - y_hat_clip))

    # BCE tests
    assert bce_loss >= 0, "BCE loss must be at least zero! no negatives!"
    assert np.isclose(bce_loss, manual_bce), "NeuralNetwork obj BCE not equal to manual BCE"
    assert nn_obj._binary_cross_entropy(np.ones(5), np.ones(5)) < 1e-4, "Identical y_hat and y must return zero error"

    with pytest.raises(ValueError): # check for unequal length
        y_hat = np.array([0])
        nn_obj._binary_cross_entropy(y, y_hat)
    with pytest.raises(ValueError): # check for empty arrays
        y_hat = np.array([])
        nn_obj._binary_cross_entropy(y, y_hat)
    with pytest.raises(ValueError): # check for non-binary inputs for y
        nn_obj._binary_cross_entropy(np.arange(5), np.arange(5))
    with pytest.raises(ValueError): # check for wrong range inputs for y_hat
        nn_obj._binary_cross_entropy(y, np.arange(4))

def test_binary_cross_entropy_backprop():
    """
    Test cases for BCE backprop.
    """
    # use example y_hat and y
    y = np.array([0, 1, 0, 0])
    y_hat = np.array([0.05, 0.96, 0.34, 0.87])
    
    bce_grad = nn_obj._binary_cross_entropy_backprop(y, y_hat)
    # clip for manual calc
    y_hat_clip = np.clip(y_hat, 1e-6, 1 - 1e-6) 
    manual_calc = -y/y_hat_clip + (1-y)/(1-y_hat_clip)

    # BCE tests
    assert np.allclose(bce_grad, manual_calc), "NeuralNetwork obj BCE backprop not equal to manual calculations"
    assert bce_grad.shape == y_hat.shape, "Gradients should equal number of outputs."
    assert not np.any(np.isnan(bce_grad)), "Gradients should not be NaNs" 

    with pytest.raises(ValueError): # check for unequal length
        y_hat = np.array([0])
        nn_obj._binary_cross_entropy_backprop(y, y_hat)
    with pytest.raises(ValueError): # check for empty arrays
        y_hat = np.array([])
        nn_obj._binary_cross_entropy_backprop(y, y_hat)
    with pytest.raises(ValueError): # check for non-binary inputs for y
        nn_obj._binary_cross_entropy_backprop(np.arange(5), np.arange(5))
    with pytest.raises(ValueError): # check for wrong range inputs for y_hat
        nn_obj._binary_cross_entropy_backprop(y, np.arange(4))

def test_mean_squared_error():
    """
    Test cases for MSE.
    """
    # use example y_hat and y
    y = np.array([0.1, 0.5, 0.3, 0.2])
    y_hat = np.array([0.1, 0.51, 0.33, 0.2])
    
    mse_loss = nn_obj._mean_squared_error(y, y_hat)
    manual_mse = np.mean((y - y_hat)**2)

    # MSE tests
    assert mse_loss >= 0, "MSE loss must be at least zero! no negatives!"
    assert np.isclose(mse_loss, manual_mse), "NeuralNetwork obj MSE not equal to manual MSE"
    assert nn_obj._mean_squared_error(np.arange(5)/5, np.arange(5)/5) == 0.0, "Identical y_hat and y must return zero error"

    with pytest.raises(ValueError): # check for unequal length
        y_hat = np.array([0])
        nn_obj._mean_squared_error(y, y_hat)
    with pytest.raises(ValueError): # check for empty arrays
        y_hat = np.array([])
        nn_obj._mean_squared_error(y, y_hat)

def test_mean_squared_error_backprop():
    """
    Test cases for MSE_backprop.
    """
    # use example y_hat and y
    y = np.array([0.1, 0.5, 0.3, 0.2])
    y_hat = np.array([0.1, 0.51, 0.33, 0.2])
    
    mse_grad = nn_obj._mean_squared_error_backprop(y, y_hat)
    manual_calc = -2*(y-y_hat)# / len(y)

    # MSE backprop tests
    assert np.allclose(mse_grad, manual_calc), "NeuralNetwork obj MSE not equal to manual MSE"
    assert mse_grad.shape == y_hat.shape, "Gradients should equal number of outputs." 
    assert not np.any(np.isnan(mse_grad)), "Gradients should not be NaNs"

    with pytest.raises(ValueError): # check for unequal length
        y_hat = np.array([0])
        nn_obj._mean_squared_error_backprop(y, y_hat)
    with pytest.raises(ValueError): # check for empty arrays
        y_hat = np.array([])
        nn_obj._mean_squared_error_backprop(y, y_hat)

def test_sample_seqs():
    """
    Test cases for balancing samples.
    """
    # imbalanced dataset
    seqs = ['AAA', 'TTT', 'CCC', 'GGG', 'ACG', 'TAC', 'GTA']
    labels = [1, 1, 1, 1, 1, 0, 0]

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # TODO: check balanced
    assert sum(sampled_labels) == len(sampled_labels) - sum(sampled_labels), "Classes should be balanced after sampling!"
    # check lengths match
    assert len(sampled_seqs) == len(sampled_labels), "Sequences and labels must be same length!"
    # check originals preserved
    assert all(s in sampled_seqs for s in seqs), "All original sequences should be present!"
    # check no new sequences invented
    assert all(s in seqs for s in sampled_seqs), "Sampled seqs should only contain sequences from original!"

def test_one_hot_encode_seqs():
    """
    Test cases for one hot encoding.
    """
    # known encoding to check against
    seqs = ['ATCG', 'AAAA']
    encodings = one_hot_encode_seqs(seqs)

    # NOTE: originally outputing array but now a list of lists
    # check shape: (n_seqs, 4 * seq_len)
    assert encodings.shape == (2, 16), "Encoding shape incorrect!"
    # check known values
    assert np.allclose(encodings[0], [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]), "ATCG encoding incorrect!"
    # check binary (only 0s and 1s)
    assert np.isin(encodings[0], [0,1]).all(), "Encodings should only contain 0s and 1s!"
    # check each base sums to 1 (one-hot property)
    assert np.all(encodings.reshape(-1, 4).sum(axis=1) == 1), "Each base should have exactly one 1 in its encoding!"
