import numpy as np
from vi_numpy import Var_Ineq
from multiprocessing import Pool
np.seterr(divide='ignore', invalid='ignore')
fresult, frecord = 'data/result.log', 'data/record.log'
print_preci = 3
hyper_paras = np.array([
    [3000000, 2e-1],
    [3000000, 1e-1],
    [6000000, 2e-2],
])[[0]]
def arr2str(arr, preci): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:.{preci}e}'}).replace('\n', '').replace(' ', '')


class var_ineq_F:
    def __init__(self, weights, biases, activation):
        self.wba = weights, biases
        self.activation = [np.tanh, (lambda x: 1 / (1 + np.exp(-x))), (lambda x: np.maximum(0, x)), (lambda x: x)][activation]
        self.grad_activation = [(lambda z: 1-np.tanh(z)**2), (lambda z: (lambda s: s*(1-s))(1/(1+np.exp(-z)))), (lambda z: (z > 0).astype(float)), (lambda z: np.ones_like(z))][activation]

    def value_Joca(self, x):
        h = x
        forward_cache = [h]
        for W, b in zip(*self.wba):
            h_ = np.dot(W, h)+b
            forward_cache.append(h_)
            h = self.activation(h_)
        J = np.eye(len(x))
        weights, _ = self.wba
        for i in reversed(range(len(weights))):
            sigma_grad = self.grad_activation(forward_cache[i])
            J = np.dot(J, sigma_grad[None, :]*weights[i])
        return h, J


def test(seed, verbose=0):
    np.random.seed(seed)
    nn_layer = [n, *hidden_layer, n]
    weights = [np.random.rand(nn_layer[i+1], nn_layer[i])-0.5 for i in range(len(nn_layer)-1)]
    biases = [np.random.rand(nn_layer[i+1])-0.5 for i in range(len(nn_layer)-1)]
    VI_problem = var_ineq_F(weights, biases, 0)
    # np.random.seed()
    init_sigma = np.random.dirichlet(np.ones(n))
    for i in range(len(hyper_paras)):
        result, success, nit, record = Var_Ineq(VI_problem).solve(
            init_sigma, canosect_TOL=1e-5, maxnit=hyper_paras[i, 0], canosect_stepln=hyper_paras[i, 1],
            verbose=verbose, record_file=frecord, print_preci=print_preci)
        with open(fresult, 'a') as fio:
            fio.writelines(f"|{seed:^4d}|{repr(success):7}|{nit:^7d}|{'|'.join([arr2str(item, print_preci) for item in record])}|{i:^10d}|\n")
        if success:
            break
    cano_sect, sigma = result


n = 100
hidden_layer = [50]
if __name__ == '__main__':
    print(f"Check results at {fresult} and iteration process at {frecord}.")
    open(fresult, 'w')
    with open(fresult, 'a') as fio:
        print_len = 6+print_preci
        fio.writelines(f"|{'seed':^4}|{'success':^7}|{'nit':^7}|{'cano_sect':^{print_len}}|{'pd_bias':^{print_len}}|{'grad':^{print_len}}|{'hyper_para':^10}|\n")
    if not (parallel_test := False):
        seed = 0
        test(seed, verbose=1)
    else:
        testrange = np.arange(2000)
        # np.random.shuffle(testrange)
        with Pool(processes=8) as pool:
            pool.map(test, testrange, chunksize=1)
