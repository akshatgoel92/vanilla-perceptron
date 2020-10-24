# Import packages
import matplotlib.pyplot as plt
import numpy as np
import os



def get_data(input_folder, data_type): 
    '''
    -------------------------------------
    Get the mean squared error loss given 
    a set of labels and model predictions.
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


def flatten_imgs(imgs):
    '''
    -------------------------------------
    Get the mean squared error loss given 
    a set of labels and model predictions.
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
    img_index = range(imgs.shape[-1])
    img_dim = imgs.shape[0]*imgs.shape[1]
    flat_imgs = np.array([imgs[:, :, img].reshape(img_dim, ) for img in img_index])

    return flat_imgs


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
    return -1*sum(y*np.log(y_pred) +(1-y)*np.log(1-y_pred))


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
    return sigmoid(x)*(1-sigmoid(x)) 


def get_finite_differences(f, x, h):
    '''
    -----------------------------------
    Returns the gradient of function f
    between point x and point x + h using
    the finite differences method.
    -----------------------------------
    Parameters
    ----------
    f: Function to differentiate
    x: Point at which to differentiate function
    h: Size of interval over which to calculate gradient

    Outputs: 
    -----------
    derivative: float or array of calculated derivatives
    -----------------------------------
    '''
    return (f(x + h)  - f(x))/h 


def get_loss_plot(output_path, train_loss, test_loss):
    '''
    -----------------------------------
    Saves plots of epoch vs. loss on
    the training and validation tests
    respectively. 
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
    
    # Calculate the no. of epochs training ran for
    # Assumes that 1 training_loss entry is 1 epoch
    epoch = range(1, len(training_loss) + 1)
    
    # Add lines for each dataset
    plt.plot(epoch, train_loss, 'r--')
    plt.plot(epoch, test_loss, 'b-')
    
    # Add annotations
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Save
    plt.savefig(output_path)


   
def sgd_update(w, alpha, grad):
    '''
    -----------------------------------
    Calculates updated value of weights
    given current weightsm learning rate, 
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
    velcotyi, momentum parameter.
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


def get_minibatch(data, offset, batch_size):
    '''
    -----------------------------------
    Divides given data into minibatches.
    -----------------------------------
    Parameters
    ----------
    data: Dataset to divide
    offset: Point to start from
    batch_size: Size of each minibatch
    
    Outputs: 
    -----------
    Batched dataset where each batch is of
    size batch_size
    -----------
    '''
    return data[offset:offset+batch_size]
    

if __name__ == '__main__':
    imgs, labels = get_data(input_folder = 'data', data_type = 'train')
    flat_imgs = flatten_imgs(imgs = imgs)
