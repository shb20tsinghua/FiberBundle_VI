# fmt: off
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from multiprocessing import Pool
from vi_numpy import Var_Ineq
from game_numpy import Var_Ineq_game
# fmt: on
np.seterr(divide='ignore', invalid='ignore', over='ignore')
print_preci = 6
current_dir = os.path.dirname(os.path.abspath(__file__))
frecord = os.path.join(current_dir, 'data\\record.log')


class nn_F:
    def __init__(self, seed, nn_layer, activation):
        np.random.seed(seed)
        self.ws = [np.random.rand(nn_layer[i+1], nn_layer[i])-0.5 for i in range(len(nn_layer)-1)]
        self.bs = [np.random.rand(nn_layer[i+1])-0.5 for i in range(len(nn_layer)-1)]
        self.activation = [np.tanh, (lambda x: 1/(1+np.exp(-x))), (lambda x: x)][activation]
        self.grad_activation = [(lambda z: 1-np.tanh(z)**2), (lambda z: (lambda s: s*(1-s))(1/(1+np.exp(-z)))), (lambda z: np.ones_like(z))][activation]

    def value_Jaco(self, x):
        h = x.flatten()
        forward_cache = [h]
        for W, b in zip(self.ws, self.bs):
            h_ = np.dot(W, h)+b
            forward_cache.append(h_)
            h = self.activation(h_)
        J = np.eye(len(h))
        for i in reversed(range(len(self.ws))):
            sigma_grad = self.grad_activation(forward_cache[i+1])
            J = np.dot(J, sigma_grad[:, None]*self.ws[i])
        return h.reshape(x.shape), J.reshape(x.shape+x.shape)


class payoff_F:
    def __init__(self, seed, m, n, r):
        np.random.seed(seed)
        self.payoff = np.random.normal(loc=1.0, scale=1/np.sqrt(m), size=(r, m, n, m))
        self.path = np.einsum_path('rij,rip,rjqi->ipjq', np.empty((r, m, m)), np.empty((r, m, n)), np.empty((r, m, n, m)), optimize='optimal')[0]

    def value_Jaco(self, pi):
        m = pi.shape[0]
        Upi = np.matmul(self.payoff.transpose(0, 1, 3, 2), pi[..., None]).squeeze(-1)
        Upi_prod = np.prod(Upi, axis=1)
        Upi_ii = Upi.diagonal(axis1=1, axis2=2)
        U_ii = self.payoff.diagonal(axis1=1, axis2=3).transpose(0, 2, 1)
        F = ((Upi_prod/Upi_ii)[:, :, None]*U_ii).sum(axis=0)
        c = Upi_prod[:, :, None]/(Upi_ii[:, :, None]*Upi.transpose(0, 2, 1))
        c[:, np.arange(m), np.arange(m)] = 0
        J = np.einsum('rij,rip,rjqi->ipjq', c, U_ii, self.payoff, optimize=self.path)
        return F, J


def result_string(seed, result, record_list): return f"|{seed:^5d}|{repr(result is not None):7}|{'|'.join([f'{item:^7d}' if type(item) is int else f'{item:.{print_preci}e}' for item in record_list])}|\n"


def test_vi(seed, n, hidden_layer, verbose=0):
    VI_problem = nn_F(seed, [n, *hidden_layer, n], -1)

    np.random.seed(seed)
    sigma_init = np.random.dirichlet(np.ones(n))
    result_sigma, record_list = Var_Ineq(VI_problem).solve(sigma_init, gap_tol=1e-5, maxnit=50000, maxsubnit=10, verbose=verbose, record_file=frecord, print_preci=print_preci)
    return result_string(seed, result_sigma, record_list)


def test_game(seed, dimension, hidden_layer, verbose=0):
    m, n = dimension
    if not (payoff_game := True):
        Game_problem = nn_F(seed, [m*n, *hidden_layer, m*n], 0)
    else:
        Game_problem = payoff_F(seed, m, n, hidden_layer[0])

    np.random.seed(seed)
    pi_init = np.random.dirichlet(np.ones(n), size=(m,))
    result_pi, record_list = Var_Ineq_game(Game_problem).solve(pi_init, gap_tol=1e-5, maxnit=50000, maxsubnit=10, verbose=verbose, record_file=frecord, print_preci=print_preci)
    return result_string(seed, result_pi, record_list)


def group_test(test_fun, dimension, hidden_layer, testrange):
    def write_result(result):
        with open(fresult, 'a') as f:
            f.write(result)
    fresult = os.path.join(current_dir, f'data\\result.log')
    print(f"Check results at {fresult} and iteration process at {frecord}.")
    with open(fresult, 'w') as fio:
        print_len = 6+print_preci
        fio.writelines(f"|{'seed':^5}|{'success':^7}|{'nit':^7}|{'gap':^{print_len}}|{'tau_sum':^{print_len}}|{'G_norm':^{print_len}}|{'avoid_n':^7}|\n")
    with Pool(process_num) as pool:
        async_results = [pool.apply_async(test_fun, args=(i, dimension, hidden_layer), callback=write_result) for i in testrange]
        [res.wait() for res in async_results]


process_num = 1
if __name__ == '__main__':
    m = 16
    n = 16
    hidden_layer = [50]
    testrange = np.arange(10)
    # group_test(test_game, (m, n), hidden_layer, testrange)
    # group_test(test_vi, n, hidden_layer, testrange)
    seed = 0
    test_game(seed, (m, n), hidden_layer, verbose=1)
    # test_vi(seed, n, hidden_layer, verbose=1)
