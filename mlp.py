import numpy as np
import pickle

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def logit(x):
    return 1 / (1 + np.exp(-x))

def logit_derivative(x):
    s = logit(x)
    return s * (1 - s)

class MLP:
    def __init__(self, layer_dims=[], act_funcs=[], regs=[], use_bn=[]):
        """
        layer_dims: a list, including the number of neurons in each layer
        """
        self.parameters = {}
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.act_funcs = act_funcs
        self.regs = regs
        self.use_bn = use_bn

        # Initialize weights and biases using Xavier
        for l in range(1, len(layer_dims)):
            xavier_scale = np.sqrt(2. / (layer_dims[l-1] + layer_dims[l]))
            self.parameters[f'W{l}'] = xavier_scale * np.random.randn(layer_dims[l], layer_dims[l-1])
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
            if l < len(layer_dims) and len(use_bn) >= l and use_bn[l-1]:
                self.parameters[f'gamma{l}'] = np.ones((layer_dims[l], 1))
                self.parameters[f'beta{l}'] = np.zeros((layer_dims[l], 1))
                self.parameters[f'running_mean{l}'] = np.zeros((layer_dims[l], 1))
                self.parameters[f'running_var{l}'] = np.ones((layer_dims[l], 1))

        # Storing the linear outputs and activation values of each layer
        self.cache = {}
        self.bn_cache = {}
        self.mode = 'train'  # or 'test'
        self.epsilon = 1e-5
        self.momentum = 0.95

    def set_mode(self, mode):
        """Set the mode to 'train' or 'test'"""
        self.mode = mode

    def batch_norm_forward(self, Z, l):
        """Forward pass for batch normalization"""
        gamma = self.parameters[f'gamma{l}']
        beta = self.parameters[f'beta{l}']
        
        if self.mode == 'train':
            mu = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)
            
            Z_norm = (Z - mu) / np.sqrt(var + self.epsilon)
            
            out = gamma * Z_norm + beta
            
            self.parameters[f'running_mean{l}'] = self.momentum * self.parameters[f'running_mean{l}'] + (1 - self.momentum) * mu
            self.parameters[f'running_var{l}'] = self.momentum * self.parameters[f'running_var{l}'] + (1 - self.momentum) * var
            
            self.bn_cache[f'Z{l}'] = Z
            self.bn_cache[f'mu{l}'] = mu
            self.bn_cache[f'var{l}'] = var
            self.bn_cache[f'Z_norm{l}'] = Z_norm
            
        elif self.mode == 'test':
            mu = self.parameters[f'running_mean{l}']
            var = self.parameters[f'running_var{l}']
            Z_norm = (Z - mu) / np.sqrt(var + self.epsilon)
            out = gamma * Z_norm + beta
            
        return out

    def batch_norm_backward(self, dout, l):
        """Backward pass for batch normalization"""
        gamma = self.parameters[f'gamma{l}']
        Z = self.bn_cache[f'Z{l}']
        mu = self.bn_cache[f'mu{l}']
        var = self.bn_cache[f'var{l}']
        Z_norm = self.bn_cache[f'Z_norm{l}']
        m = Z.shape[1]
        
        dgamma = np.sum(dout * Z_norm, axis=1, keepdims=True)
        dbeta = np.sum(dout, axis=1, keepdims=True)
        
        dZ_norm = dout * gamma
        
        dvar = np.sum(dZ_norm * (Z - mu) * -0.5 * (var + self.epsilon)**(-1.5), axis=1, keepdims=True)
        
        dmu = np.sum(dZ_norm * -1 / np.sqrt(var + self.epsilon), axis=1, keepdims=True) + \
              dvar * np.sum(-2 * (Z - mu), axis=1, keepdims=True) / m
        
        dZ = dZ_norm / np.sqrt(var + self.epsilon) + \
             dvar * 2 * (Z - mu) / m + \
             dmu / m
        
        return dZ, dgamma, dbeta

    def forward(self, X):
        A = X.T
        self.cache['A0'] = A

        for l in range(1, self.L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            Z = np.dot(W, A) + b

            if l < self.L and len(self.use_bn) >= l and self.use_bn[l-1]:
                Z = self.batch_norm_forward(Z, l)

            self.cache[f'Z{l}'] = Z

            if l < self.L:
                if self.act_funcs[l - 1] == 'relu':
                    A = relu(Z)
                if self.act_funcs[l - 1] == 'logit':
                    A = logit(Z)
            else:
                A = Z
            self.cache[f'A{l}'] = A

        return A.T
    
    def cross_entropy_loss(self, AL, y):
        N = AL.shape[0]
        AL -= np.max(AL, axis=1, keepdims=True)
        exp_AL = np.exp(AL)
        softmax_probs = exp_AL / np.sum(exp_AL, axis=1, keepdims=True)
        log_softmax_probs = -np.log(softmax_probs[np.arange(N), y])
        loss = np.sum(log_softmax_probs) / N

        l2_loss = 0.0
        for l in range(1, self.L + 1):
            W = self.parameters[f'W{l}']
            l2_loss += 0.5 * self.regs[l - 1] * np.sum(W ** 2)
        loss += l2_loss

        dAL = softmax_probs.copy()
        dAL[np.arange(N), y] -= 1
        dAL /= N

        return loss, dAL


    def backward(self, dAL, Y):
        m = Y.shape[0]
        grads = {}
        dA = dAL.T
        for l in reversed(range(1, self.L + 1)):
            A_prev = self.cache[f'A{l-1}']
            Z = self.cache[f'Z{l}']
            W = self.parameters[f'W{l}']

            if l < self.L:
                if self.act_funcs[l - 1] == 'relu':
                    dZ = dA * relu_derivative(Z)
                if self.act_funcs[l - 1] == 'logit':
                    dZ = dA * logit_derivative(Z)

                if len(self.use_bn) >= l and self.use_bn[l-1]:
                    dZ, dgamma, dbeta = self.batch_norm_backward(dZ, l)
                    grads[f"dgamma{l}"] = dgamma
                    grads[f"dbeta{l}"] = dbeta
            else:
                dZ = dA
            
            dW = np.dot(dZ, A_prev.T) + self.regs[l - 1] * W
            db = np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                dA = np.dot(W.T, dZ)
            
            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db

        return grads
    
    def predict(self, X):
        prev_mode = self.mode
        self.set_mode('test')
        scores = self.forward(X)
        self.set_mode(prev_mode)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
    
    def save_model(self, save_path):
        param_list = [self.layer_dims, self.act_funcs, self.use_bn]
        for l in range(1, self.L + 1):
            layer_params = {
                f'W{l}': self.parameters[f'W{l}'],
                f'b{l}': self.parameters[f'b{l}'],
                'reg': self.regs[l - 1]
            }
            if l < self.L and len(self.use_bn) >= l and self.use_bn[l-1]:
                layer_params.update({
                    f'gamma{l}': self.parameters[f'gamma{l}'],
                    f'beta{l}': self.parameters[f'beta{l}'],
                    f'running_mean{l}': self.parameters[f'running_mean{l}'],
                    f'running_var{l}': self.parameters[f'running_var{l}']
                })
            param_list.append(layer_params)
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
    
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)
        self.layer_dims = param_list[0]
        self.act_funcs = param_list[1]
        self.use_bn = param_list[2]
        self.L = len(self.layer_dims) - 1
        self.parameters = {}
        self.regs = []

        for l in range(1, self.L + 1):
            layer_params = param_list[l + 2]  # +2 because first 3 elements are config
            self.parameters[f'W{l}'] = layer_params[f'W{l}']
            self.parameters[f'b{l}'] = layer_params[f'b{l}']
            self.regs.append(layer_params['reg'])
            
            if l < self.L and len(self.use_bn) >= l and self.use_bn[l-1]:
                self.parameters[f'gamma{l}'] = layer_params[f'gamma{l}']
                self.parameters[f'beta{l}'] = layer_params[f'beta{l}']
                self.parameters[f'running_mean{l}'] = layer_params[f'running_mean{l}']
                self.parameters[f'running_var{l}'] = layer_params[f'running_var{l}']

        self.cache = {}
        self.bn_cache = {}

    def load_model_without_BN(self, load_path):
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)
        self.layer_dims = param_list[0]
        self.act_funcs = param_list[1]
        self.L = len(self.layer_dims) - 1
        self.parameters = {}
        self.regs = []

        for l in range(1, self.L + 1):
            layer_params = param_list[l + 1]
            self.parameters[f'W{l}'] = layer_params[f'W{l}']
            self.parameters[f'b{l}'] = layer_params[f'b{l}']
            self.regs.append(layer_params['reg'])

        self.cache = {}


        

            
