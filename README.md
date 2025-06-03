# AutoGrad Clone

A quick project made by me in a week end to learn more about gradient descend and backpropagation.
Allows for a fast deployment of linear layers and relu activations.

A test file is provided comparing a model implemented with the custom autograd to a pytorch implementation.

## Example model implementation:

```
class YourModel(ModelArch):
    def __init__(self):
        self.layers = {
            #Some example layers with kaiming normal weight initialisation
            "fc1": Linear(10, 8, weights=he_normal_init(10, 8)),
            "fc2": Linear(8, 3, weights=he_normal_init(8, 3))
        }
        super().__init__()

    def forward(self, x):
        # Forward pass implmentation
        out = ReLU(self.layers["fc1"](x))
        return self.layers["fc2"](out)
```
 A clear training loop implementation is displayed in test.py.
