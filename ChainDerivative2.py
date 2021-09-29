import sys
sys.path.append('C:\\Users\\Shashwat Ratna\\Desktopr\\realshh\\Learn\\Deep_Learning')
import Derivative
import numpy as np
from typing import List,Callable

def sigmoid(x:np.ndarray):
    '''
    Apply the sigmoid function to each element in the input ndarray
    '''
    return 1/(1+np.exp(-x))

Chain=[Callable[[np.ndarray],np.ndarray],Callable[[np.ndarray],np.ndarray]]


def chain_deriv_2(chain:Chain,input_range:np.ndarray)->np.ndarray:
    '''
    Uses chain rule to compute the derivative of two nested functions
    (f2(f1(x)))'=f2'(f1(x))*f1'(x)
    '''

    assert len(chain)==2,\
        "This function requires 'Chain' objects of length 2"

    assert input_range.ndim==1,\
        "Function requires a one dimensional ndarray as input_range"
    
    f1=chain[0]
    f2=chain[1]

    #f1(x)
    f1_of_x=f1(input_range)

    #df1/du
    df1dx=Derivative.deriv(f1,input_range)

    #df2/d(f1(x))
    df2du=Derivative.deriv(f2,f1(input_range))

    #Multiplying
    return df1dx*df2du

PLOT_RANGE=np.arange(-3,3,0.01)
chain_2=[sigmoid,np.square]
chain_1=[np.square,sigmoid]

print(chain_deriv_2(chain_2,PLOT_RANGE),":",PLOT_RANGE)


