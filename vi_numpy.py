import numpy as np
from scipy.optimize import brentq


class Var_Ineq:
    def __init__(self, VI_F):
        self.VI_F = VI_F

    def print_log(self, nit):
        with open(self.record_file, 'a') as fio:
            fio.writelines(f"|{nit:^7d}|{'|'.join([f'{item:.{self.print_preci}e}' for item in self.record_list])}|\n")

    def solve(self, sigma_init, preci, stepsize=1e-1, maxnit=50000, verbose=0, record_file=None, print_preci=3):
        self.nit, self.maxnit, self.mu_check_TOL, self.eta = 0, maxnit, preci, stepsize
        self.verbose, self.record_file, self.print_preci = verbose if record_file else 0, record_file, print_preci
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                print_len = 6+print_preci
                fio.writelines(f"|{'nit':^7}|{'mu_check':^{print_len}}|{'mu_sum':^{print_len}}|{'G_norm':^{print_len}}|{'step_norm':^{print_len}}|\n")
        try:
            sigma = self.path_following(sigma_init, sigma_init*1e3)
        except UserWarning as err:
            print(err)
            return None, self.nit, self.record_list
        else:
            return sigma, self.nit, self.record_list

    def path_following(self, sigma, mu):
        beta = 1
        n = len(sigma)
        while True:
            try:
                correct_result = self.corrector_iter(sigma, mu)
            except (ValueError, np.linalg.LinAlgError) as err:
                if self.verbose >= 1:
                    with open(self.record_file, 'a') as fio:
                        fio.writelines(f"{repr(err)}\n")
                correct_result = None
            if self.verbose >= 1:
                self.print_log(self.nit)
            if correct_result is not None:
                sigma, J_sigma, mu_check = correct_result
                if mu_check < self.mu_check_TOL:
                    return sigma
                smJ_bkp = [item.copy() for item in [sigma, mu, J_sigma]]
                beta *= 1-self.eta
            else:
                sigma, mu, J_sigma = [item.copy() for item in smJ_bkp]
                beta += 1
                mu += 1e-1*beta*mu.sum()*sigma
            mu_last = mu.copy()
            mu = (mu_last*(1-self.eta)).clip(min=self.mu_check_TOL/(n+1))
            I_1sigma = np.eye(n)-np.kron(np.ones(n)[:, None], sigma[None, :])
            predictor_sigma = np.linalg.solve(J_sigma+mu_last.sum()*np.eye(n), I_1sigma.dot((mu_last-mu)/sigma))
            _predictor_sigma = predictor_sigma/max(np.linalg.norm(predictor_sigma), 1)
            sigma = (lambda vec: vec/vec.sum())(np.exp(np.log(sigma)-_predictor_sigma))

    def corrector_comp(self, sigma, mu, standard_newton=False):
        n = len(sigma)
        mu_sum = mu.sum()
        F, FJ = self.VI_F.value_Joca(sigma)
        r_difference = F-F.min()
        I_1sigma = np.eye(n)-np.kron(np.ones(n)[:, None], sigma[None, :])
        G_tilde_sigma_mu = I_1sigma.dot(F-mu/sigma)
        G_sigma_mu = sigma*G_tilde_sigma_mu
        J_sigma = I_1sigma.dot(FJ*sigma[None, :]+np.eye(n)*np.dot(I_1sigma, F)[None, :]).dot(I_1sigma)
        J_G = J_sigma+mu_sum*np.eye(n)
        if standard_newton:
            corrector_sigma = np.linalg.solve(J_G.T.dot(J_G)+min(1, np.linalg.norm(G_sigma_mu)/n)*np.eye(n), J_G.T.dot(G_tilde_sigma_mu))
        else:
            r = r_difference+brentq(lambda v: (mu/(r_difference+v)).sum()-1, 0, mu_sum)
            sigma_bias = sigma-mu/r
            corrector_sigma = np.linalg.solve(J_G.T.dot(np.diag(sigma/r)).dot(J_G)+min(1, np.linalg.norm(G_sigma_mu)/n)*np.eye(n), J_G.T.dot(sigma_bias))
        corrector_sigma = I_1sigma.dot(corrector_sigma)
        self.record_list = [sigma.dot(r_difference), mu_sum, np.linalg.norm(G_sigma_mu), np.linalg.norm(corrector_sigma)]
        return corrector_sigma, J_sigma, self.record_list[0], self.record_list[2]

    def corrector_iter(self, sigma, mu, maxsubnit=100):
        for subnit in range(maxsubnit):
            if self.nit > self.maxnit:
                raise UserWarning("Maximum Iteration Number Exceeded!")
            corrector_sigma, J_sigma, mu_check, G_sigma_mu_norm = self.corrector_comp(sigma, mu)
            if self.verbose >= 2:
                self.print_log(subnit)
            if np.isnan(G_sigma_mu_norm).any():
                return None
            if G_sigma_mu_norm < min(1e-9, self.mu_check_TOL/(len(sigma)+1)):
                return sigma, J_sigma, mu_check
            sigma = (lambda vec: vec/vec.sum())(np.exp(np.log(sigma)-corrector_sigma))
            self.nit += 1
        return None
