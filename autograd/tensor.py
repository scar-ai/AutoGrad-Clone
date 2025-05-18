from typing import NamedTuple, Callable
import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]



def is_array(entering):
    if isinstance(entering, np.ndarray):
        return entering
    else:
        return np.array(entering)

def is_tensor(entering, requires_grad=False):
    if isinstance(entering, Tensor):
        return entering
    else:
        return Tensor(entering, requires_grad)


class Tensor:
    def __init__(self,
                 data,
                 requires_grad=False,
                 grad_depends=[]):
        
        self.data = is_array(data)
        self.requires_grad = requires_grad
        self.grad_depends = grad_depends
        
        self.shape = self.data.shape
        self.grad = None
        if self.requires_grad:
            self.zero_grad()

        

    def __repr__(self):
        return f"Tensor: {self.data}, shape={self.shape}, requires_grad={self.requires_grad}"
    
    
    def __add__(self, other):
        return AddOp(self, is_tensor(other))

    def __radd__(self, other):
        return AddOp(is_tensor(other), self)
    
    
    def __sub__(self, other):
        return AddOp(self, -1 * (is_tensor(other)))
    
    def __rsub__(self, other):
        return AddOp(is_tensor(other), -1 * self)
    
    
    def __mul__(self, other):
        return MulOp(self, is_tensor(other))
    
    def __rmul__(self, other):
        return MulOp(is_tensor(other), self)
    
    def __truediv__(self, other):
        return DivOp(self, is_tensor(other))
    
    def __rtruediv__(self, other):
        return DivOp(is_tensor(other), self)
    
    
    def exp(self):
        return ExpOp(self)


    def zero_grad(self):
        self.grad = (np.zeros_like(self.data, dtype=np.float32))

    def sum(self, axis=None, keepdims=False):
        return SumOp(self, axis, keepdims)

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "Backward can only be called on tensors that requires grad."
        
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("Grad must be specified for non-0-tensor")
        else:
            grad = is_tensor(grad)
       
        
        self.grad = self.grad + grad.data
        for dependency in self.grad_depends:
            #print(dependency.grad_fn)
            #print(dependency.tensor)

            backward_grad = dependency.grad_fn(grad)
            #print("\n")
            dependency.tensor.backward(backward_grad)



def negate_broadcasting(grad_val, x):
    ndims_added = grad_val.data.ndim - x.data.ndim

    for _ in range(ndims_added):
        grad_val = grad_val.sum(axis=0)
            
    for i, dim_size in enumerate(x.shape):
        if i < grad_val.data.ndim and dim_size == 1 and grad_val.shape[i] > 1:
            grad_val = grad_val.sum(axis=i, keepdims=True)

    if x.shape == () and grad_val.shape != ():
        grad_val = grad_val.sum()

    return grad_val


def SumOp(x1, axis=None, keepdims=False):
    result = x1.data.sum(axis = axis, keepdims=keepdims)
    require_grad = x1.requires_grad
    
    grad_depends = []

    if x1.requires_grad:
        def grad_fn(grad):
            return grad * np.ones_like(x1.data)
        grad_depends.append(Dependency(tensor=x1, grad_fn=grad_fn))

    return Tensor(data=result, 
                    requires_grad=require_grad,
                    grad_depends=grad_depends)

def AddOp(x1, x2):
    result = x1.data + x2.data
    requires_grad = x1.requires_grad or x2.requires_grad

    grad_depends = []
    
    if x1.requires_grad:
        def grad_add_fn1(grad_val):
            return negate_broadcasting(grad_val, x1)
        
        grad_depends.append(Dependency(tensor=x1, grad_fn=grad_add_fn1))


    if x2.requires_grad:
        def grad_add_fn2(grad_val):
            return negate_broadcasting(grad_val, x2)
        
        grad_depends.append(Dependency(tensor=x2, grad_fn=grad_add_fn2))
        
    return Tensor(data=result, requires_grad=requires_grad, grad_depends=grad_depends)

def MulOp(x1: 'Tensor', x2: 'Tensor') -> 'Tensor':
    result_data = x1.data * x2.data
    requires_grad = x1.requires_grad or x2.requires_grad

    grad_depends = []
   
    if x1.requires_grad:
        def grad_mul_fn1(grad_output_data):
            grad_val = grad_output_data * x2.data
            return negate_broadcasting(grad_val, x1)
        
        grad_depends.append(Dependency(tensor=x1, grad_fn=grad_mul_fn1))

    if x2.requires_grad:
        def grad_mul_fn2(grad_output_data ):
            grad_val = grad_output_data * x1.data
            return negate_broadcasting(grad_val, x2)
        
        grad_depends.append(Dependency(tensor=x2, grad_fn=grad_mul_fn2))
        
    return Tensor(data=result_data, requires_grad=requires_grad, grad_depends=grad_depends)



def DivOp(x1: 'Tensor', x2: 'Tensor') -> 'Tensor':
    result_data = np.divide(x1.data, x2.data)
    requires_grad = x1.requires_grad or x2.requires_grad

    grad_depends = []
   
    if x1.requires_grad:
        def grad_div_fn1(grad_output_data ):
            grad_val = np.divide(grad_output_data, x2.data)
            return negate_broadcasting(grad_val, x1)
        
        grad_depends.append(Dependency(tensor=x1, grad_fn=grad_div_fn1))

    if x2.requires_grad:
        def grad_div_fn2(grad_output_data ):
            grad_val = grad_output_data * (-result_data / x2.data)
            return negate_broadcasting(grad_val, x2)
        
        grad_depends.append(Dependency(tensor=x2, grad_fn=grad_div_fn2))
        
    return Tensor(data=result_data, requires_grad=requires_grad, grad_depends=grad_depends)

def MatMulOp(x1: 'Tensor', x2: 'Tensor') -> 'Tensor':
    result_data = x1.data @ x2.data
    requires_grad = x1.requires_grad or x2.requires_grad

    grad_depends = []
    if x1.requires_grad:
        def grad_matmul_fn1(grad_output_tensor: 'Tensor') -> 'Tensor':
            grad_val_np = grad_output_tensor.data @ x2.data.T
            grad_val_as_tensor = Tensor(grad_val_np, requires_grad=False) 
            return negate_broadcasting(grad_val_as_tensor, x1)
        
        grad_depends.append(Dependency(tensor=x1, grad_fn=grad_matmul_fn1))

    if x2.requires_grad:
        def grad_matmul_fn2(grad_output_tensor: 'Tensor') -> 'Tensor':
            x1_d = x1.data 
            grad_d = grad_output_tensor.data
            
            if x1_d.ndim == 1:
                grad_val_np = np.outer(x1_d, grad_d)
            else:
                grad_val_np = x1_d.T @ grad_d
            
            grad_val_as_tensor = Tensor(grad_val_np, requires_grad=False)
            return negate_broadcasting(grad_val_as_tensor, x2)
        
        grad_depends.append(Dependency(tensor=x2, grad_fn=grad_matmul_fn2))
        
    return Tensor(data=result_data, requires_grad=requires_grad, grad_depends=grad_depends)


def ExpOp(x1: 'Tensor') -> 'Tensor':
    result_data = np.exp(x1.data)
    requires_grad = x1.requires_grad

    grad_depends = []
   
    if x1.requires_grad:
        def grad_exp_fn1(grad_output_data):
            grad_val = grad_output_data * result_data

            return negate_broadcasting(grad_val, x1)
        
        grad_depends.append(Dependency(tensor=x1, grad_fn=grad_exp_fn1))
        
    return Tensor(data=result_data, requires_grad=requires_grad, grad_depends=grad_depends)



def LogOp(x1: 'Tensor') -> 'Tensor':
    result_data = np.log(x1.data)
    requires_grad = x1.requires_grad

    grad_depends = []
   
    if x1.requires_grad:
        def grad_log_fn1(grad_output_data ):
            grad_val = grad_output_data.data*(1/x1.data)
            return negate_broadcasting(grad_val, x1)
        
        grad_depends.append(Dependency(tensor=x1, grad_fn=grad_log_fn1))
        
    return Tensor(data=result_data, requires_grad=requires_grad, grad_depends=grad_depends)




def randn(
        size,
        requires_grad=False,
        grad_depends=[]):


    size = size if isinstance(size, tuple) else (size,)
    return Tensor(data = np.random.rand(*size),
                  requires_grad=requires_grad,
                  grad_depends=grad_depends)