# Imports
import pytest
import numpy as np
from regression import logreg, utils
from sklearn.preprocessing import StandardScaler

# Make sure the sigmoid is correctly implemented
def test_sigmoid_prediction():

    # Make a regressor and manually set the weights
    r = logreg.LogisticRegression(3)
    r.W = np.array([0.25, 0.5, 0.75, 1])

    # Put a known vector (including bias) through prediction function
    # and compare to hand-calculated output
    X = np.array([1, 1, 1, 1])
    computed = r.make_prediction(X)
    known = 0.92414
    assert np.allclose(computed, known), "Computed sigmoid calculation does not match known calculation."

# Make sure BCE is correctly implemented
def test_bce_loss():

    # Make a regressor and manually set the weights
    r = logreg.LogisticRegression(3)
    r.W = np.array([0.25, 0.5, 0.75, 1])

    # Create known datapoints and labels
    X = np.array([[0.2, 0.2, 0.2, 1], [0.4, 0.4, 0.4, 1], [0.6, 0.6, 0.6, 1]])
    y = np.array([0, 1, 1])

    # Compare computed BCE to hand-calculated output
    computed = r.loss_function(X, y)
    known = 0.62143
    assert np.allclose(computed, known), "Computed BCE loss does not match known loss."

# Make sure the gradient calculation is correctly implemented
def test_gradient():

    # Make a regressor and manually set the weights
    r = logreg.LogisticRegression(3)
    r.W = np.array([0.25, 0.5, 0.75, 1])

    # Create known datapoints and labels
    X = np.array([[0.2, 0.2, 0.2, 1], [0.4, 0.4, 0.4, 1], [0.6, 0.6, 0.6, 1]])
    y = np.array([0, 1, 1])

    # Compare computed gradient to hand-calculated output
    computed = r.calculate_gradient(X, y)
    known = np.array([0.00396976, 0.00396976, 0.00396976, 0.16258163])
    assert np.allclose(computed, known), "Computed gradient does not match known gradient."

# Make sure the overall training works as expected
def test_training():

    # Define features to use
    f = ['Penicillin V Potassium 500 MG',
         'Computed tomography of chest and abdomen',
         'Plain chest X-ray (procedure)',
         'Low Density Lipoprotein Cholesterol',
         'Creatinine',
         'AGE_DIAGNOSIS']

    # Load in data using constant split and scale
    X_train, X_val, y_train, y_val = utils.loadDataset(features = f, split_percent = 0.8, split_state = 42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # Create regressor and manually set starting weights (got them from a random run of main.py with same parameters)
    r = logreg.LogisticRegression(num_feats = 6, learning_rate = 1e-5, tol = 1e-6)
    starting_weights = np.array([1.59028308, 0.02014035, -0.3606881, -0.82422628, 0.2625946, -0.19699378, 1.57214908])
    r.W = starting_weights

    # Perform training
    r.train_model(X_train, y_train, X_val, y_val)

    # Make sure weights are different
    assert not np.allclose(starting_weights, r.W), "Weights have not changed during training."

    # Test that validation loss has decreased over the training period and it's below 1 at the end
    assert np.mean(r.loss_history_val[:10]) > np.mean(r.loss_history_val[-10:]), "Validation loss has not decreased over training."
    assert r.loss_history_val[-1] < 1, "Final validation loss is higher than 1 -- something wrong with training?"

    # Compute accuracy on the validation set and make sure it's better than chance
    X_val_with_bias = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    predicted = np.round(r.make_prediction(X_val_with_bias))
    assert np.sum(predicted == y_val) / len(y_val) > 0.5, "Final accuracy is worse than chance!"