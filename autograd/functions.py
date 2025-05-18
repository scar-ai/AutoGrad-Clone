from autograd.tensor import *


def softmax(x:'Tensor'):
    return x.exp()/(x.exp()).sum()


def CrossEntropyLoss(x:'Tensor', y:'Tensor'):
    return -1*((LogOp(softmax(x))*y).sum())

def ReLU(x:'Tensor'):
    requires_grad = x.requires_grad  
    grad_depends = []

    if requires_grad:
        def grad_fn(grad_output_data):
            grad_val = grad_output_data * (x.data > 0).astype(float)
            return negate_broadcasting(grad_val, x)

        grad_depends.append(Dependency(tensor=x, grad_fn=grad_fn))

    output = Tensor(np.maximum(0, x.data), requires_grad, grad_depends=grad_depends)
    return output


def he_normal_init(input_size, output_size):
    fan_in = input_size
    std_dev = np.sqrt(2.0 / fan_in)

    weights = np.random.normal(loc=0.0, scale=std_dev, size=(input_size, output_size))
    return weights


class Linear:
    def __init__(self, input_size, output_size, weights=None, bias=None):
        if weights is None:
            weights = randn((input_size, output_size), requires_grad=True)
        else:
            weights = is_tensor(weights, requires_grad=True)

        if bias is None:
            bias = randn((output_size), requires_grad=True)
        else:
            bias = is_tensor(bias, requires_grad=True)

        self.parameters = {
            "weights": is_tensor(weights),
            "bias": is_tensor(bias)}
        
        
        
    def __call__(self, x):
        return self.forward(x)


    def _update_weights(self, lr):
        self.parameters["weights"] -= lr*self.parameters["weights"].grad
        self.parameters["bias"] -= lr*self.parameters["bias"].grad


    def forward(self, x):
        x = is_tensor(x)
        return (MatMulOp(x, self.parameters["weights"])) + self.parameters["bias"]


class ModelArch:
    def __init__(self):
        self._register_parameters(self.layers.values())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _register_parameters(self, layers):
        self.parameters = []
        for layer in layers:
            if hasattr(layer, "parameters"):
                self.parameters.append(layer)

    def apply(self, function):
        assert self.parameters, "The model must contain parameters before applying a function to the model's weights"

        for param in self.parameters:
            function(param.parameters["weights"])


    def forward(self, *args, **kwargs):
        pass



class SGD:
    def __init__(self, lr, model:ModelArch):
        assert lr, "You need to specify a learning rate for the optimizer"
        assert model, "You need to specify a model"
        self.lr = lr
        self.model = model

    def step(self):
        for layer in self.model.parameters:
            layer._update_weights(self.lr)

    def zero_grad(self):
        for layer in self.model.parameters:
            for param in layer.parameters.values():
                param.zero_grad()
                for dependency in param.grad_depends:
                    dependency.tensor.zero_grad()
                    dependency.tensor.grad_depends = []
                param.grad_depends = []
