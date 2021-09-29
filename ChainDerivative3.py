import BasicFunctionsNumPy
from typing import List,Callable
import Derivative
import numpy as np
Chain=List[Callable]

def chain_deriv_3(chain:Chain,input_range:np.ndarray)->np.ndarray:
    """
    Computes the derivative of 3 layer compsition functions
    """
    assert len(chain)==3,\
        "Provide three functions"
    assert input_range.ndim==1,\
        "Input array must be one dimensional"
    
    f1=chain[0]
    f2=chain[1]
    f3=chain[2]

    f1_of_x=f1(input_range)
    df1dx=Derivative.deriv(f1,input_range)
    df2dx=Derivative.deriv(f2,f1_of_x)
    df3dx=Derivative.deriv(f3,f2(input_range))
    return df1dx*df2dx*df3dx

def main():
    random_range=np.arange(-3,3,0.001)
    chain1=[BasicFunctionsNumPy.sigmoid,BasicFunctionsNumPy.square,BasicFunctionsNumPy.leaky_relu]
    #print(chain_deriv_3(chain1,random_range))
    pom=np.savetxt("chain_deriv_3",chain_deriv_3(chain1,random_range))
    

if __name__=="__main__":
    main()
