import numpy as np
from scipy.optimize.elementwise import find_root


class Var_Ineq_game:
    def __init__(self, VI_F):
        self.VI_F = VI_F

    def solve(self, pi_init, gap_tol, maxnit=50000, maxsubnit=10, verbose=0, record_file=None, print_preci=6):
        self.nit, self.maxnit, self.maxsubnit, self.gap_TOL = 0, maxnit, maxsubnit, gap_tol
        self.verbose, self.record_file, self.print_preci = verbose if record_file else 0, record_file, print_preci
        self.m, self.n, self.mn = *pi_init.shape, pi_init.shape[0]*pi_init.shape[1]
        self.tau_min = min(1e-9, self.gap_TOL/self.mn)
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                print_len = 6+print_preci
                fio.writelines(f"|{'nit':^7}|{'gap':^{print_len}}|{'tau_sum':^{print_len}}|{'G_norm':^{print_len}}|{'d_sign':^7}|{'subnit':^7}|{'tang_stepln':^{print_len}}|{'avoid_n':^7}|\n")
        try:
            pi = self.path_following(pi_init)
        except UserWarning as err:
            print(err)
            pi = None
        return pi, [self.record_list[i] for i in [0, 1, 2, 3, 7]]

    def print_record(self):
        with open(self.record_file, 'a') as fio:
            fio.writelines(f"|{'|'.join([f'{item:^7d}' if type(item) is int else f'{item:.{self.print_preci}e}' for item in self.record_list])}|\n")

    def path_following(self, pi):
        subnit_low, subnit_high = int(self.maxsubnit*0.3), int(self.maxsubnit*0.6)
        self.tangent_stepln, singu_avoid_times = 1e-1, 0
        self.record_list = [0]*8
        corrected, subnit, pi, tau_isum, tau_check_isum, dtau_sign = self.predictor_corrector(pi, 1e3*np.ones(self.m))
        while True:
            if (tau_check_isum <= self.gap_TOL).all():
                return pi
            if not corrected:
                tau_isum = self.singu_avoid(tau_isum, tau_check_isum)
                stepln_adjust = 0.5
                singu_avoid_times += 1
            else:
                stepln_adjust = 1.5 if subnit < subnit_low else 1 if subnit >= subnit_low and subnit < subnit_high else 0.5
            self.record_list[5:8] = [subnit, self.tangent_stepln, singu_avoid_times]
            if self.verbose >= 1:
                self.print_record()
            corrected, subnit, pi, tau_isum, tau_check_isum, dtau_sign_new = self.predictor_corrector(pi, tau_isum)
            if dtau_sign_new != dtau_sign:
                stepln_adjust = 0.9
            dtau_sign = dtau_sign_new
            self.tangent_stepln *= stepln_adjust

    def singu_avoid(self, tau_isum, tau_check_isum):
        tol = 0.1
        w_tilde_new = np.random.dirichlet(np.ones(self.m))
        tau_sum_max = (1+2e-2)*tau_isum.sum()
        tau_isum_new = w_tilde_new*((1-tol)*tau_sum_max-tau_check_isum.sum())+tau_check_isum+tol/self.m*tau_sum_max
        return tau_isum_new

    def state_comp(self, pi):
        F, JF = self.VI_F.value_Jaco(pi)
        r_difference = F-F.min(axis=-1, keepdims=True)
        tau_check_isum = np.vecdot(pi, r_difference, axis=-1)
        I_1pi = np.eye(self.n)[None, :, :].repeat(self.m, axis=0)-pi[:, None, :].repeat(self.n, axis=1)
        _I_1pi = (I_1pi[:, None, :, :]*np.eye(self.m)[:, :, None, None]).transpose(0, 2, 1, 3).reshape(self.mn, self.mn)
        J_pi = _I_1pi.dot(JF.reshape(self.mn, self.mn)*pi.flatten()[None, :]+np.diag(np.matvec(I_1pi, F).flatten())).dot(_I_1pi)
        return F, r_difference, tau_check_isum, I_1pi, J_pi

    def corrector_comp(self, pi, tau, newton1=False):
        F, r_difference, tau_check_isum, I_1pi, J_pi = self.state_comp(pi)
        tau_isum = tau.sum(axis=-1)
        G_pi_tau_norm = np.linalg.norm(np.matvec(I_1pi.swapaxes(1, 2), pi*F-tau).flatten())
        J_G = J_pi+np.diag(tau_isum.repeat(self.n))
        if newton1:
            G_tilde_pi_tau = np.matvec(I_1pi, F-tau/pi).flatten()
            corrector_pi = np.linalg.solve(J_G.T.dot(J_G)+min(1, G_pi_tau_norm/self.mn)*np.eye(self.mn), J_G.T.dot(G_tilde_pi_tau/max(1, np.linalg.norm(G_tilde_pi_tau))))
        else:
            r = r_difference+find_root(lambda v, index: (tau[index]/(r_difference[index]+v[:, None])).sum(axis=-1)-1, (np.zeros(self.m), tau_isum), args=(np.arange(self.m),)).x[:, None]
            corrector_pi = np.linalg.solve(J_G.T.dot(np.diag((pi/r).flatten())).dot(J_G)+min(1, G_pi_tau_norm/self.mn)*np.eye(self.mn), J_G.T.dot((pi-tau/r).flatten()))
        corrector_pi = np.matvec(I_1pi, corrector_pi.reshape(self.m, self.n))
        return corrector_pi, G_pi_tau_norm, tau_check_isum

    def predictor_corrector(self, pi, tau_isum, decrease_only=False):
        F, r_difference, tau_check_isum, I_1pi, J_pi = self.state_comp(pi)
        state_bkp = [item.copy() for item in [pi, tau_check_isum]]
        r = r_difference+(tau_isum-tau_check_isum)[:, None]
        tau = pi*r

        J_G = J_pi+np.diag(tau_isum.repeat(self.n))
        dtau_sign = 1 if decrease_only else np.linalg.slogdet(J_G)[0]
        tangent_pi = np.linalg.solve(J_G, np.matvec(I_1pi, r).flatten())
        tangent_ln = np.sqrt(1+tangent_pi.dot(tangent_pi))
        eta = min(0.1, self.tangent_stepln/tangent_ln)
        self.tangent_stepln = eta*tangent_ln

        tau = (tau*(1-eta*dtau_sign)).clip(min=self.tau_min)
        tau_isum_new = tau.sum(axis=-1)
        tau_isum_new_max = tau_isum_new.max()
        pi = (lambda vec: vec/vec.sum(axis=-1, keepdims=True))(np.exp(np.log(pi)-tangent_pi.reshape((self.m, self.n))*eta*dtau_sign))

        for subnit in range(self.maxsubnit):
            if self.nit > self.maxnit:
                raise UserWarning("Maximum Iteration Number Exceeded!")
            corrector_pi, G_pi_tau_norm, tau_check_isum = self.corrector_comp(pi, tau)
            self.record_list[:5] = [self.nit, tau_check_isum.max(), tau_isum_new_max, G_pi_tau_norm, int(dtau_sign)]
            if self.verbose >= 2:
                self.print_record()
            if (corrected := G_pi_tau_norm <= self.tau_min):
                break
            pi = (lambda vec: vec/vec.sum(axis=-1, keepdims=True))(np.exp(np.log(pi)-corrector_pi))
            self.nit += 1
        if not corrected:
            pi, tau_check_isum, tau_isum_new = *state_bkp, tau_isum
        return corrected, subnit, pi, tau_isum_new, tau_check_isum, dtau_sign
