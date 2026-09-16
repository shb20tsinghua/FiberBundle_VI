import numpy as np
from scipy.optimize import brentq


class Var_Ineq:
    def __init__(self, VI_F):
        self.VI_F = VI_F

    def solve(self, sigma_init, gap_tol, maxnit=50000, maxsubnit=10, verbose=0, record_file=None, print_preci=6):
        self.nit, self.maxnit, self.maxsubnit, self.gap_TOL = 0, maxnit, maxsubnit, gap_tol
        self.verbose, self.record_file, self.print_preci = verbose if record_file else 0, record_file, print_preci
        self.n = len(sigma_init)
        self.mu_min = min(1e-9, self.gap_TOL/self.n)
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                print_len = 6+print_preci
                fio.writelines(f"|{'nit':^7}|{'gap':^{print_len}}|{'mu_sum':^{print_len}}|{'G_norm':^{print_len}}|{'d_sign':^7}|{'subnit':^7}|{'tang_stepln':^{print_len}}|{'avoid_n':^7}|\n")
        try:
            sigma = self.path_following(sigma_init)
        except UserWarning as err:
            print(err)
            sigma = None
        return sigma, [self.record_list[i] for i in [0, 1, 2, 3, 7]]

    def print_record(self):
        with open(self.record_file, 'a') as fio:
            fio.writelines(f"|{'|'.join([f'{item:^7d}' if type(item) is int else f'{item:.{self.print_preci}e}' for item in self.record_list])}|\n")

    def path_following(self, sigma):
        subnit_low, subnit_high = int(self.maxsubnit*0.3), int(self.maxsubnit*0.6)
        self.tangent_stepln, singu_avoid_times = 1e-1, 0
        self.record_list = [0]*8
        corrected, subnit, sigma, mu_sum, mu_check_sum, dmu_sign = self.predictor_corrector(sigma, 1e3)
        while True:
            if mu_check_sum <= self.gap_TOL:
                return sigma
            if not corrected:
                mu_sum = self.singu_avoid(mu_sum, mu_check_sum)
                stepln_adjust = 0.5
                singu_avoid_times += 1
            else:
                stepln_adjust = 1.5 if subnit < subnit_low else 1 if subnit >= subnit_low and subnit < subnit_high else 0.5
            self.record_list[5:8] = [subnit, self.tangent_stepln, singu_avoid_times]
            if self.verbose >= 1:
                self.print_record()
            corrected, subnit, sigma, mu_sum, mu_check_sum, dmu_sign_new = self.predictor_corrector(sigma, mu_sum)
            if dmu_sign_new != dmu_sign:
                stepln_adjust = 0.9
            dmu_sign = dmu_sign_new
            self.tangent_stepln *= stepln_adjust

    def singu_avoid(self, mu_sum, mu_check_sum):
        mu_sum_new = (1+2e-2)*mu_sum
        return mu_sum_new

    def state_comp(self, sigma):
        F, JF = self.VI_F.value_Jaco(sigma)
        r_difference = F-F.min()
        mu_check_sum = sigma.dot(r_difference)
        I_1sigma = np.eye(self.n)-sigma[None, :].repeat(self.n, axis=0)
        J_sigma = I_1sigma.dot(JF*sigma[None, :]+np.diag(np.dot(I_1sigma, F))).dot(I_1sigma)
        return F, r_difference, mu_check_sum, I_1sigma, J_sigma

    def corrector_comp(self, sigma, mu, newton1=False):
        F, r_difference, mu_check_sum, I_1sigma, J_sigma = self.state_comp(sigma)
        mu_sum = mu.sum()
        G_sigma_mu_norm = np.linalg.norm(I_1sigma.T.dot(sigma*F-mu))
        J_G = J_sigma+mu_sum*np.eye(self.n)
        if newton1:
            G_tilde_sigma_mu = I_1sigma.dot(F-mu/sigma)
            corrector_sigma = np.linalg.solve(J_G.T.dot(J_G)+min(1, G_sigma_mu_norm/self.n)*np.eye(self.n), J_G.T.dot(G_tilde_sigma_mu/max(1, np.linalg.norm(G_tilde_sigma_mu))))
        else:
            r = r_difference+brentq(lambda v: (mu/(r_difference+v)).sum()-1, 0, mu_sum)
            corrector_sigma = np.linalg.solve(J_G.T.dot(np.diag(sigma/r)).dot(J_G)+min(1, G_sigma_mu_norm/self.n)*np.eye(self.n), J_G.T.dot(sigma-mu/r))
        corrector_sigma = I_1sigma.dot(corrector_sigma)
        return corrector_sigma, G_sigma_mu_norm, mu_check_sum

    def predictor_corrector(self, sigma, mu_sum, decrease_only=False):
        F, r_difference, mu_check_sum, I_1sigma, J_sigma = self.state_comp(sigma)
        state_bkp = [item.copy() for item in [sigma, mu_check_sum]]
        r = r_difference+mu_sum-mu_check_sum
        mu = sigma*r

        J_G = J_sigma+mu_sum*np.eye(self.n)
        dmu_sign = 1 if decrease_only else np.linalg.slogdet(J_G)[0]
        tangent_sigma = np.linalg.solve(J_G, I_1sigma.dot(r))
        tangent_ln = np.sqrt(1+tangent_sigma.dot(tangent_sigma))
        eta = min(0.1, self.tangent_stepln/tangent_ln)
        self.tangent_stepln = eta*tangent_ln

        mu = (mu*(1-eta*dmu_sign)).clip(min=self.mu_min)
        mu_sum_new = mu.sum()
        sigma = (lambda vec: vec/vec.sum())(np.exp(np.log(sigma)-tangent_sigma*eta*dmu_sign))

        for subnit in range(self.maxsubnit):
            if self.nit > self.maxnit:
                raise UserWarning("Maximum Iteration Number Exceeded!")
            corrector_sigma, G_sigma_mu_norm, mu_check_sum = self.corrector_comp(sigma, mu)
            self.record_list[:5] = [self.nit, mu_check_sum, mu_sum_new, G_sigma_mu_norm, int(dmu_sign)]
            if self.verbose >= 2:
                self.print_record()
            if (corrected := G_sigma_mu_norm <= self.mu_min):
                break
            sigma = (lambda vec: vec/vec.sum())(np.exp(np.log(sigma)-corrector_sigma))
            self.nit += 1
        if not corrected:
            sigma, mu_check_sum, mu_sum_new = *state_bkp, mu_sum
        return corrected, subnit, sigma, mu_sum_new, mu_check_sum, dmu_sign
