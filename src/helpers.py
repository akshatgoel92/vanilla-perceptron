# Import packages
#import cv2
import matplotlib.pyplot as plt
import numpy as np
import os



def get_data(input_folder, data_type): 
    '''
    -------------------------------------
    Get imgs and  label given input_folder
    and data_type
    -------------------------------------
    Parameters
    ----------
    input_folder: Path to input folder [rel. to current]
    data_type: train, dev, or test set
    
    Outputs: 
    -----------
    Numpy array of images and labels respectively
    -----------
    '''
    imgs_path = os.path.join(input_folder, 'fashion-{}-imgs.npz'.format(data_type))
    labels_path = os.path.join(input_folder, 'fashion-{}-labels.npz'.format(data_type))

    imgs = np.load(imgs_path)
    labels = np.load(labels_path)
    
    return imgs, labels


def load_all_data(input_folder):
    '''
    -------------------------------------
    Load all the train, dev and test arrays.
    -------------------------------------
    Parameters
    ----------
    input_folder

    Outputs:
    -----------
    Images and labels for train, test, dev
    -----------
    '''
    X_train, y_train = get_data(input_folder, "train")
    X_dev, y_dev = get_data(input_folder, "dev")
    X_test, y_test = get_data(input_folder, "test")

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def prep_data(data_path):
    '''
    --------------------
    Prepare data for 
    training
    --------------------
    Parameters: 
    data_path: Where the data is
    --------------------
    Output: 
    X_train_flattened: Flattened training features
    X_dev_flattened: Flattened dev. features
    X_test_flattened: Flattened test features
    y_train: Flattened training labels
    y_dev: Flattened training labels
    y_test: Flattened training labels
    --------------------
    '''
    # Load
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_all_data(data_path)
    
    # Flatten
    X_train_flattened = vectorized_flatten(X_train)
    X_dev_flattened = vectorized_flatten(X_dev)
    X_test_flattened = vectorized_flatten(X_test)
    
    # Reshape labels
    y_train = y_train.reshape(1, -1)
    y_dev = y_dev.reshape(1, -1)
    y_test = y_test.reshape(1, -1)
    
    # Return
    return(X_train_flattened, X_dev_flattened, X_test_flattened, y_train, y_dev, y_test)


def vectorized_flatten(imgs):
    '''
    -------------------------------------
    Given a list of imgs. each of n X n
    return a list of imgs each of n**2 X 1
    -------------------------------------
    Parameters
    ----------
    imgs: list of imgs
    
    Outputs: 
    -----------
    MSE value for given preds.
    -----------
    '''
    return np.ravel(imgs).reshape((imgs.shape[0]*imgs.shape[1], imgs.shape[-1]))


def get_mse_loss(y,y_pred):
    '''
    -------------------------------------
    Get the mean squared error loss given 
    a set of labels and model predictions.
    -------------------------------------
    Parameters
    ----------
    y: Data labels
    y_pred: Model predictions
    
    Outputs: 
    -----------
    MSE value for given preds.
    -----------
    '''
    return np.sum((y-y_pred)**2/2)


def get_log_loss(y, y_pred):
    '''
    -------------------------------------
    Get log likelihood loss given a set
    of labels and model predictions.
    -------------------------------------
    Parameters
    ----------
    y: Data labels
    y_pred: Model predictions
    
    Outputs: 
    -----------
    Log loss value for given preds.
    -----------
    '''
    epsilon = 1e-7
    return -1*np.sum(y*np.log(y_pred+epsilon) +(1-y)*np.log(1-y_pred+epsilon))


def sigmoid(x):
    '''
    -----------------------------------
    Calculates sigmoid activation value at x.
    -----------------------------------
    Parameters
    ----------
    x: Point at which to calculate sigmoid
       value.
    
    Outputs: 
    -----------
    Sigmoid value at x
    -----------
    '''
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    '''
    -----------------------------------
    Calculates sigmoid derivative 
    value at x.
    -----------------------------------
    Parameters
    ----------
    x: Point at which to calculate
       derivative.
    
    Outputs: 
    -----------
    Sigmoid derivative at x
    -----------
    '''

    s = sigmoid(x)

    return s * (1-s)



def plot_loss(output_path, train_loss, label='Training Loss'):
    '''
    -----------------------------------
    Saves plots of epoch vs. loss on
    the training and validation tests
    respectively depending on the label.
    -----------------------------------
    Parameters
    ----------
    output_path: Path where plots should be saved
    train_loss: Array of training losses
    test_loss: Array of testing losses
    
    Outputs: 
    -----------
    Plot of training and testing loss by epoch 
    saved at output_path
    -----------
    '''
    epoch = range(1, len(train_loss) + 1)
    
    # Start a new figure
    plt.clf()
    
    # Add lines for each dataset
    plt.plot(epoch, train_loss, 'r--')

    # Add annotations
    plt.legend([label])
    plt.xlabel('Epoch')
    plt.ylabel(label)

    # Save
    plt.savefig(output_path)


def get_accuracy(target, pred):
        '''
    -----------------------------------
    Calculates correct classification rate
    between target vector [labels] 
    and predictions.
    -----------------------------------
    Parameters
    ----------
    target: vector of labels
    pred: vector of predictions
    These should be the same shape
    
    Outputs: 
    -----------
    Correct classification rate between 
    target and pred
    -----------
    '''
    return np.sum(target==pred)/max(target.shape)


def gradient_update(w, alpha, grad):
    '''
    -----------------------------------
    Calculates updated value of weights
    given current weights, learning rate, 
    gradient.
    -----------------------------------
    Parameters
    ----------
    w: current weights
    alpha: learning rate
    grad: gradient
    
    Outputs: 
    -----------
    New weights after applying update rule
    -----------
    '''
    return w - np.multiply(alpha,grad)



def sgd_with_momentum_update(w, alpha, grad, velocity, momentum) :
    '''
    -----------------------------------
    Calculates updated value of weights using
    momentum rule given current weights, 
    learning rate, gradient, current 
    velocity, momentum parameter.
    -----------------------------------
    Parameters
    ----------
    w: current weights
    alpha: learning rate
    grad: gradient
    velocity: current velocity
    momentum: momentum parameter
    
    Outputs: 
    -----------
    New weights after applying momentum update rule
    New velocity to use at next update
    -----------
    '''
    new_velocity = np.multiply(momentum, velocity) - np.multiply(alpha, grad)
    w_new = w + new_velocity
    return w_new, new_velocity


def get_best_epoch(history):
    '''
    --------------------
    Get best training epoch
    --------------------
    Parameters: 
    history: dictionary which
    contains record of metrics
    at every epoch of a training
    run
    --------------------
    Output: 
    best_epoch: 
    best_accuracy:
    best_loss: 
    --------------------
    '''
    # Store results
    best_epoch = np.array(history["losses"]).argmin()
    best_accuracy = history['accuracies'][best_epoch]
    best_loss = history['losses'][best_epoch]
    
    # Display results
    print(f"best accuracy: {history['accuracies'][best_epoch]}")
    print(f"best loss: {history['losses'][best_epoch]}")
    print(f"best epoch: {best_epoch}")
    
    return(best_epoch, best_accuracy, best_loss)


def get_best_dev_epoch(history):
    '''
    --------------------
    Get best dev epoch
    --------------------
    Parameters: 
    history: dictionary which
    contains record of metrics
    at every epoch of a training
    run
    --------------------
    Output: 
    best_epoch: minimum dev. loss epoch
    best_accuracy: accuracy on best_epoch
    best_loss: loss on best_epoch
    --------------------
    '''
    # Store results
    best_epoch = np.array(history["dev_loss"]).argmin()
    best_accuracy = history['dev_accuracies'][best_epoch]
    best_loss = history['dev_loss'][best_epoch]
    
    # Display results
    print(f"best dev accuracy: {history['dev_accuracies'][best_epoch]}")
    print(f"best dev loss: {history['dev_loss'][best_epoch]}")
    print(f"best dev epoch: {best_epoch}")
    
    return(best_epoch, best_accuracy, best_loss)


def get_results(X_dev, y_dev, history, best_epoch, label="dev"):
    '''
    --------------------
    Get accuracy on dev set
    --------------------
    Parameters: 
    X_dev: Dev. set features
    y_dev: Dev. set labels
    history: history object
    best_epoch: epoch for which to get weights and biases
    --------------------
    Output: 
    accuracy: predicted accuracy on dev. set
    --------------------
    '''
    w = history["weights"][best_epoch]
    b = history["biases"][best_epoch]
    activations = forward_pass(X_dev, w, b)

    y_dev_prob = activations[-1]
    y_dev_pred = np.where(y_dev_prob > 0.5, 1, 0)

    loss = get_log_loss(y_dev, y_dev_prob)
    accuracy = get_accuracy(y_dev, y_dev_pred)
    print(f"{label} set accuracy: {accuracy}")
    
    return(accuracy)


def shuffle_data(X, y):
    '''
    --------------------
    Shuffle a dataset
    --------------------
<<<<<<< HEAD
    Parameters:
    weights: Current set of weights
    biases: Current set of biases
    gradients: Current set of gradients
    learning_rate: parameter to guide SGD step size
    --------------------
    Output:
    Updated weights and biases
=======
    Parameters: 
    X: Vector of features
    y: vector of labels
    --------------------
    Output: 
    X, y permuted in such a
    way as to maintain the correspondence
    between features and labels
>>>>>>> new-hyper-tuning
    --------------------
    '''
    # Data is currently unshuffled; we should shuffle
    # each X[i] with its corresponding y[i]
    perm = np.random.permutation(max(y.shape))
    X = X[: , perm]
    y = y[: , perm]

    return(X, y)


def get_best_results(history, metric='losses'):
    '''
    --------------------
    Takes in a record of
    training and returns
    epoch on which min.
    training loss
    was reached, and the
    training loss and 
    accuracy at this epoch
    --------------------
    Parameters: 
    history: record of metrics
    from a training run
    --------------------
    Output: 
    best_epoch: epoch of min. training loss
    best_accuracy: accuracy at best_epoch
    best_loss: loss at best_epoch
    --------------------
    '''
    # Store results
    if "loss" in metric:
        best_epoch = np.array(history[metric]).argmin()
    else:
        best_epoch = np.array(history[metric]).argmax()

    best_accuracy = history['accuracies'][best_epoch]
    best_loss = history['losses'][best_epoch]
    
    # Display results
    print(f"training accuracy at best epoch: {history['accuracies'][best_epoch]}")
    print(f"training loss at best epoch: {history['losses'][best_epoch]}")
    print(f"best epoch: {best_epoch}")
    
    return(best_epoch, best_accuracy, best_loss)

