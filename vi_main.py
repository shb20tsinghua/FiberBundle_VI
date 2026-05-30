import numpy as np
from multiprocessing import Pool
from vi_numpy import Var_Ineq
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
np.seterr(divide='ignore', invalid='ignore')
print_preci = 6
current_dir = os.path.dirname(os.path.abspath(__file__))
frecord = os.path.join(current_dir, 'data\\record.log')


class var_ineq_F:
    def __init__(self, weights, biases, activation):
        self.wba = weights, biases
        self.activation = [np.tanh, (lambda x: 1/(1+np.exp(-x))), (lambda x: x)][activation]
        self.grad_activation = [(lambda z: 1-np.tanh(z)**2), (lambda z: (lambda s: s*(1-s))(1/(1+np.exp(-z)))), (lambda z: np.ones_like(z))][activation]

    def value_Joca(self, x):
        h = x
        forward_cache = [h]
        for W, b in zip(*self.wba):
            h_ = np.dot(W, h)+b
            forward_cache.append(h_)
            h = self.activation(h_)
        J = np.eye(len(x))
        for i in reversed(range(len(self.wba[0]))):
            sigma_grad = self.grad_activation(forward_cache[i+1])
            J = np.dot(J, sigma_grad[:, None]*self.wba[0][i])
        return h, J


def test(seed, n, hidden_layer, verbose=0):
    np.random.seed(seed)
    nn_layer = [n, *hidden_layer, n]
    weights = [np.random.rand(nn_layer[i+1], nn_layer[i])-0.5 for i in range(len(nn_layer)-1)]
    biases = [np.random.rand(nn_layer[i+1])-0.5 for i in range(len(nn_layer)-1)]
    VI_problem = var_ineq_F(weights, biases, 0)

    # np.random.seed(0)
    sigma_init = np.random.dirichlet(np.ones(n))
    result_sigma, nit, record_list = Var_Ineq(VI_problem).solve(
        sigma_init, preci=1e-5, stepsize=1e-1, maxnit=50000,
        verbose=verbose, record_file=frecord, print_preci=print_preci)
    return f"|{seed:^5d}|{repr(result_sigma is not None):7}|{nit:^7d}|{'|'.join([f'{item:.{print_preci}e}' for item in record_list])}|\n"


def group_test(n, hidden_layer, testrange):
    def write_result(result):
        with open(fresult, 'a') as f:
            f.write(result)
    fresult = os.path.join(current_dir, f'data\\result.log')
    print(f"Check results at {fresult} and iteration process at {frecord}.")
    with open(fresult, 'w') as fio:
        print_len = 6+print_preci
        fio.writelines(f"|{'seed':^5}|{'success':^7}|{'nit':^7}|{'mu_check':^{print_len}}|{'mu_sum':^{print_len}}|{'G_norm':^{print_len}}|{'step_norm':^{print_len}}|\n")
    with Pool(process_num) as pool:
        async_results = [pool.apply_async(test, args=(i, n, hidden_layer), callback=write_result) for i in testrange]
        [res.wait() for res in async_results]
    with open(fresult, 'r') as f:
        lines = f.readlines()
        nits = np.array([int(line[15:22]) for line in lines[1:]])
    with open(os.path.join(current_dir, 'data\\nits.log'), 'a') as fio:
        fio.writelines(f"{n}, {int(np.mean(nits))}, {int(np.median(nits))}\n")


process_num = 1
if __name__ == '__main__':
    n = 100
    hidden_layer = [50]
    testrange = np.arange(5)
    group_test(n, hidden_layer, testrange)
