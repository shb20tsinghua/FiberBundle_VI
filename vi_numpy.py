import numpy as np
from scipy.optimize import brentq
def arr2str(arr, preci): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:.{preci}e}'}).replace('\n', '').replace(' ', '')


class Var_Ineq:
    def __init__(self, var_ineq_F):
        self.var_ineq_F = var_ineq_F

    def solve(self, sigma, canosect_TOL=1e-5, maxnit=500000, canosect_stepln=2e-1, verbose=0, record_file=None, print_preci=3):
        init_mu, grad_stepln, tangent_stepln, maxsubnit = 1e3, 2e-4, 2e-2, 10000
        mu = sigma*init_mu
        verbose = verbose if record_file else 0
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                print_len = 6+print_preci
                fio.writelines(f"|{'nit':^7}|{'cano_sect':^{print_len}}|{'pd_bias':^{print_len}}|{'grad':^{print_len}}|\n")
        nit, pre_grad, eta, _beta = 0, 0, tangent_stepln, 1.0
        while True:
            subnit, adam_m, adam_v = 0, 0, 0
            while True:
                grad, sigma_bias, r, cano_sect, comat = self.onto_equilbundl(mu, sigma)
                record_list = [np.linalg.norm(item, np.inf) for item in [cano_sect, sigma_bias, grad]]
                cano_sect_norm, sigma_biasnorm, grad_norm = record_list

                if verbose >= 2:
                    with open(record_file, 'a') as fio:
                        fio.writelines(f"|{subnit:^7d}|{'|'.join([arr2str(item, print_preci) for item in record_list])}|\n")

                if (nit := nit+1) > maxnit:
                    return [cano_sect, sigma], False, nit, record_list
                elif (subnit := subnit+1) > maxsubnit or sigma_biasnorm < 1e-9 or np.linalg.norm(pre_grad-(pre_grad := grad), np.inf) < 1e-12:
                    if sigma_biasnorm < 3e-8:
                        eta += (tangent_stepln-eta)*0.1
                        _beta *= 0.2
                        if cano_sect_norm < canosect_TOL:
                            return [cano_sect, sigma], True, nit, record_list
                    else:
                        if not sigma_biasnorm < 1e-6:
                            mu, sigma, r = bpv_bkp
                        eta *= 0.5
                        _beta += 1
                    break

                adam_m += 1e-1*(grad-adam_m)
                adam_v += 1e-3*(grad**2-adam_v)
                dsigma = grad_stepln*adam_m/(adam_v+4*np.finfo(float).eps)**0.5
                sigma = (lambda vec: vec/vec.sum())(np.exp(np.log(sigma)-dsigma))

            bpv_bkp = mu.copy(), sigma.copy(), r.copy()
            diff, beta = self.along_equilbundl(sigma, r, comat, _beta*cano_sect.sum())
            mu = cano_sect+beta*sigma
            dsigma = sigma*diff.sum(axis=-1)
            _canosect_stepln = (eta/np.linalg.norm(dsigma)).clip(max=canosect_stepln)
            mu *= 1-_canosect_stepln
            sigma = (lambda vec: vec/vec.sum())(np.exp(np.log(sigma)-_canosect_stepln*dsigma))
            if verbose >= 1:
                with open(record_file, 'a') as fio:
                    fio.writelines(f"|{nit:^7d}|{'|'.join([arr2str(item, print_preci) for item in record_list])}|{eta:^.{print_preci}e}|{beta:^+.{print_preci}e}|{_beta:^.{print_preci}e}|\n")

    def onto_equilbundl(self, mu, sigma):
        n = len(sigma)
        F, FJ = self.var_ineq_F.value_Joca(sigma)
        min_F, _cano_sect = (lambda _min_F: (_min_F, F-_min_F))(F.min())
        v = brentq(lambda v: (mu/(_cano_sect-v)).sum()-1, -n*mu.max(), 0)+min_F
        r = F-v
        sigma_bias = sigma-mu/r
        comat = FJ*sigma[None, :]+np.eye(n)*r[None, :]
        grad = (lambda grad: grad-grad.sum()*sigma)(np.dot(sigma_bias[None, :], comat)[0, :])
        return grad, sigma_bias, r, sigma*_cano_sect, comat

    def along_equilbundl(self, sigma, r, comat, _beta):
        n = len(sigma)
        coff = np.eye(n)-np.kron(np.ones(n)[:, None], sigma[None, :])
        comat_ = np.dot(coff, comat)
        rmin = r.min()
        beta = -rmin*np.random.rand()*0e0+_beta+1e-8
        diff = np.linalg.solve(comat_+beta*np.eye(n), coff*(r+beta)[None, :])
        return diff, beta+rmin
