# Import packages
import matplotlib.pyplot as plt
import numpy 



def get_data(): 
    '''
    ----------------------
    Input: 
    Output: 

    Description: 
    -----------------------
    '''
    pass


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
    derivative = (f(x + h)  - f(x))/h 

    return(derivative)


def get_loss_plot(train_loss, test_loss):
    '''
    -----------------------------------
    Saves plots of epoch vs. loss on
    the training and validation tests
    respectively. 
    -----------------------------------
    Parameters
    ----------
    train_loss: Array of training losses
    test_loss: Array of testing losses
    
    Outputs: 
    -----------
    Plot of training and testing loss by epoch
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
    
    # Show and save
    plt.show()
    plt.savefig(output_path)


def main():
    '''
    ----------------------
    Input: 
    Output: 

    Description: 
    -----------------------
    '''
    pass


if __name__ = '__main__':
    pass