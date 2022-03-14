Python version=python 3.6

## Instructions to run UNF code
- To run the code, use ``python3 main.py`` and pass appropriate options.

### Options details
- hidden_layers: Number of hidden layers
- hidden_nodes: Number of hidden nodes (Should be a list of length hidden_layers)
- total_epochs: Number of training epochs
- std_a: Standard deviation of top layer initialization
- batch_size
- initialization:
    - gaussian-sqrt(m): Initialize hidden layer weights with zero-mean Gaussian and 1/m variance where m=number of hidden nodes
    - gaussian-std: Initializes hidden layer weights using standard Gaussian
- base_distribution
- dataset_file: Specify relative path of dataset file to read. Each line in dataset should have a data point with space separated elements of dimension of that data point.
- results_file: location of folder where generated files will be saved 
- integrate_method: choices=["right-sum", "clenshaw-curtis"]
- int_steps: Number of quadrature points in approximation
- learning_rate
- seed: Manual seed to fix randomness in training
