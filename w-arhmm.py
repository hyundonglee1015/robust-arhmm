import numpy as np
from scipy.stats import norm, invgamma, multivariate_normal, multinomial, dirichlet
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from copy import deepcopy
from numpy.linalg import inv, multi_dot, matrix_rank
import matplotlib.pyplot as plt
import numpy.random as npr
from autograd.tracer import getval
from autograd.scipy.linalg import solve_triangular
import time
#npr.seed(0)

def lse(a):
    return np.exp(a - logsumexp(a, keepdims=True))

class single_ARHMM(object):
    def __init__(self, D, Q, sigma2_y0=1e2, a1=4e1, sigma2_b=1e2, sigma2_A=1e-3,
                 sigma2_y=1e2, a=1, c=1, RHO = -1):
        self.D = D
        self.Q = Q
        self.Sigma_y0 = sigma2_y0 * np.eye(D)
        self.a1 = a1

        self.Sigma_b = sigma2_b * np.eye(D)

        self.Sigma_A = sigma2_A * np.eye(D)

        self.sigma2_y = sigma2_y
        self.C = (np.cos(a * np.log(1 + c)) + 1) / 2
        self.a = a
        self.c = c
        self.RHO = RHO

        assert (Q < D)

        self._initialize()

    # initialize model params
    def _initialize(self):
        D, Q = self.D, self.Q
        Sigma_y0, a1, Sigma_b, Sigma_A = self.Sigma_y0, self.a1, self.Sigma_b, self.Sigma_A

        initial_log_joint = 0

        mean0_D = np.zeros(D)
        mean0_Q = np.zeros(Q)

        # Initialize latent variables: y0
        y0 = multivariate_normal.rvs(mean=mean0_D, cov=Sigma_y0)
        initial_log_joint += multivariate_normal.logpdf(y0, mean=mean0_D, cov=Sigma_y0)
        self.y0 = y0

        # Initialize latent variables: LAMBDA
        LAMBDA = invgamma.rvs(a=a1, scale=1)
        initial_log_joint += invgamma.logpdf(LAMBDA, a=a1, scale=1)
        self.LAMBDA = LAMBDA

        # initialize latent variables: b
        b = multivariate_normal.rvs(mean=mean0_D, cov=Sigma_b)
        initial_log_joint += multivariate_normal.logpdf(b, mean=mean0_D, cov=Sigma_b)
        self.b = b

        # initialize latent variables: v,u
        v = np.zeros((D, Q))
        u = np.zeros((D, Q))
        for d in range(self.D):
            Sigma_vu = LAMBDA * np.eye(Q)
            v_d = multivariate_normal.rvs(mean=mean0_Q, cov=Sigma_vu)
            initial_log_joint += multivariate_normal.logpdf(v_d, mean=mean0_Q, cov=Sigma_vu)

            u_d = multivariate_normal.rvs(mean=mean0_Q, cov=Sigma_vu)
            initial_log_joint += multivariate_normal.logpdf(u_d, mean=mean0_Q, cov=Sigma_vu)

            v[d] = deepcopy(v_d)
            u[d] = deepcopy(u_d)

        self.v = v
        self.u = u

        # initialize latent variables: A
        A = np.zeros((D, D))
        for d in range(D):
            mean = np.dot(u[d], np.transpose(v))
            A_d = multivariate_normal.rvs(mean=mean, cov=Sigma_A)
            initial_log_joint += multivariate_normal.logpdf(A_d, mean=mean, cov=Sigma_A)

            A[d] = deepcopy(A_d)

        self.A = A

        self.initial_log_joint = initial_log_joint

    # samples states and observations
    def sample(self, T):
        D = self.D
        a, c, C, sigma2_y = self.a, self.c, self.C, self.sigma2_y
        b, A = self.b, self.A

        bA = np.concatenate((b[:, np.newaxis], A), axis=1)

        y = np.zeros((D, T))
        log_joint = 0

        for t in range(T):
            phi_t = self.compute_phi_t(y, t)
            #print(phi_t)
            mean = np.dot(bA, phi_t)
            y[:, t] = multivariate_normal.rvs(mean=mean, cov=sigma2_y * np.eye(D))
            log_joint += multivariate_normal.logpdf(y[:, t], mean=mean, cov=sigma2_y * np.eye(D))

        return y, self.initial_log_joint + log_joint

    def compute_initial_log_joint(self, y, phi):
        D, T = self.D, y.shape[1]
        y0, b, A = self.y0, self.b, self.A
        Sigma_y = self.sigma2_y * np.eye(T)

        bA = np.concatenate((b[:, np.newaxis], A), axis=1)

        log_joint = 0
        for d in range(D):
            mean = np.dot(bA[d], phi)
            log_joint += multivariate_normal.logpdf(y[d], mean=mean, cov=Sigma_y)

        return self.initial_log_joint + log_joint

    def compute_phi_t(self, y, t):
        D = y.shape[0]
        T = y.shape[1]
        y0 = self.y0
        a, c, C = self.a, self.c, self.C

        RHO = self.RHO

        phi_t = np.ones(D + 1)  # (D+1,T)

        if t == 0:
            phi_t[1:] = deepcopy(C * y0)
            return phi_t
        else:
            phi_t_temp = np.zeros(D)
            for tau in range(1, t + 1):
                C_tau = a * np.log(tau + c)
                if C_tau >= RHO - np.pi and C_tau <= RHO + np.pi:
                    C_tau_cos = (np.cos(C_tau - RHO) + 1) / 2
                else:
                    C_tau_cos = 0

                phi_t_temp += C_tau_cos * y[:, t - tau]

            phi_t[1:] = deepcopy(phi_t_temp)

            return phi_t

    def compute_phi(self, y0, y):
        D, T = self.D, y.shape[1]
        a, c, C = self.a, self.c, self.C
        RHO = self.RHO

        phi = np.ones((D + 1, T))

        for t in range(T):
            if t == 0:
                phi[1:, 0] = deepcopy(C * y0)
                continue

            phi_t = np.zeros(D)

            for tau in range(1, t + 1):
                C_tau = a * np.log(tau + c)
                if C_tau >= RHO - np.pi and C_tau <= RHO + np.pi:
                    C_tau_cos = (np.cos(C_tau - RHO) + 1) / 2
                else:
                    C_tau_cos = 0
                phi_t += C_tau_cos * y[:, t - tau]

            phi[1:, t] = deepcopy(phi_t)

        return phi

    def _sample_y0(self, Sigma_y0, C, A, Sigma_y1, b, y1):
        Sigma_y1_inv = inv(Sigma_y1)
        A_t = np.transpose(A)

        cov = inv(Sigma_y0 + C ** 2 * multi_dot([A_t, Sigma_y1_inv, A]))
        mean = C * (multi_dot([A_t, Sigma_y1_inv, y1]) - multi_dot([A_t, Sigma_y1_inv, b]))
        mean = np.dot(cov, mean)  # np.transpose(np.dot(cov, mean))[0]

        y0 = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_y0 = multivariate_normal.logpdf(y0, mean=mean, cov=cov)

        return y0, log_joint_y0

    def _sample_v_d(self, A_d, Sigma_A, u, LAMBDA, Q):
        Sigma_v = LAMBDA * np.eye(Q)
        u_t = np.transpose(u)
        Sigma_A_inv = inv(Sigma_A)

        cov = inv(inv(Sigma_v) + multi_dot([u_t, Sigma_A_inv, u]))
        mean = np.dot(multi_dot([A_d, Sigma_A_inv, u]), cov)

        v_d = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_v_d = multivariate_normal.logpdf(v_d, mean=mean, cov=cov)

        return v_d, log_joint_v_d

    def _sample_u_d(self, A_d, Sigma_A, v, LAMBDA, Q):
        Sigma_u = LAMBDA * np.eye(Q)
        v_t = np.transpose(v)
        Sigma_A_inv = inv(Sigma_A)

        cov = inv(inv(Sigma_u) + multi_dot([v_t, Sigma_A_inv, v]))
        mean = np.dot(multi_dot([A_d, Sigma_A_inv, v]), cov)

        u_d = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_u_d = multivariate_normal.logpdf(u_d, mean=mean, cov=cov)

        return u_d, log_joint_u_d

    def _sample_bA_d(self, M, Sigma_bA, y_d, Sigma_y, phi):
        Sigma_bA_inv = inv(Sigma_bA)
        Sigma_y_inv = inv(Sigma_y)
        phi_t = np.transpose(phi)

        cov = inv(Sigma_bA_inv + multi_dot([phi, Sigma_y_inv, phi_t]))
        mean = np.dot(np.dot(M, Sigma_bA_inv) + multi_dot([y_d, Sigma_y_inv, phi_t]), cov)

        bA_d = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_bA_d = multivariate_normal.logpdf(bA_d, mean=mean, cov=cov)

        return bA_d, log_joint_bA_d

    # run gibbs sampler
    def run_gibbs_sampler(self, y, iterations=100):
        D, Q, T = self.D, self.Q, y.shape[1]
        y0, LAMBDA, b = deepcopy(self.y0), deepcopy(self.LAMBDA), deepcopy(self.b)
        u, v, A = deepcopy(self.u), deepcopy(self.v), deepcopy(self.A)
        Sigma_y0, a1, Sigma_b, Sigma_A = self.Sigma_y0, self.a1, self.Sigma_b, self.Sigma_A
        Sigma_y1, Sigma_y = self.sigma2_y * np.eye(D), self.sigma2_y * np.eye(T)
        C = self.C

        assert (D == y.shape[0])

        samples = defaultdict(list)
        log_joints = []

        samples['y0'].append(deepcopy(y0))
        samples['LAMBDA'].append(deepcopy(LAMBDA))
        samples['b'].append(deepcopy(b))
        samples['v'].append(deepcopy(v))
        samples['u'].append(deepcopy(u))
        samples['A'].append(deepcopy(A))

        # Compute PHI:
        phi = self.compute_phi(y0, y)

        log_joint = self.compute_initial_log_joint(y, phi)

        log_joints.append(log_joint)

        print('Initial log joint: {}'.format(log_joint))
        print('------------------------------------------------')

        # Concatenate b & A:
        bA = np.concatenate((b[:, np.newaxis], A), axis=1)
        vec_Sigma_bA = np.zeros(D + 1)
        vec_Sigma_bA[0] = self.Sigma_b[0, 0]
        vec_Sigma_bA[1:] = np.diag(Sigma_A)
        Sigma_bA = np.diag(vec_Sigma_bA)

        for itr in range(iterations):
            print('Iteration {}'.format(itr))
            log_joint = 0

            # SAMPLE y0
            y0, log_joint_y0 = self._sample_y0(Sigma_y0=Sigma_y0,
                                               C=C, A=bA[:, 1:], Sigma_y1=Sigma_y1, b=bA[:, 0], y1=y[:, 0])
            log_joint += log_joint_y0

            # Update phi
            phi[1:, 0] = deepcopy(C * y0)

            # SAMPLE LAMBDA
            new_a = a1 + D * Q
            new_scale = 1 + np.sum(np.add(np.square(v), np.square(u))) / 2
            LAMBDA = invgamma.rvs(a=new_a, scale=new_scale)
            log_joint += invgamma.logpdf(LAMBDA, a=new_a, scale=new_scale)

            # SAMPLE v
            for d in range(D):
                v_d, log_joint_v_d = self._sample_v_d(A_d=bA[:, d+1], Sigma_A=Sigma_A, u=u, LAMBDA=LAMBDA, Q=Q)
                log_joint += log_joint_v_d
                v[d] = deepcopy(v_d)

            # SAMPLE u
            for d in range(D):
                u_d, log_joint_u_d = self._sample_u_d(A_d=bA[d, 1:], Sigma_A=Sigma_A, v=v, LAMBDA=LAMBDA, Q=Q)
                log_joint += log_joint_u_d
                u[d] = deepcopy(u_d)

            # v_prime = v with a row padded to the top
            row = np.zeros((1, Q))
            v_prime = np.concatenate((row, v), axis=0)

            # SAMPLE bA
            for d in range(D):
                bA_d, log_joint_bA_d = self._sample_bA_d(M=np.dot(u[d], v_prime.T),
                                                         Sigma_bA=Sigma_bA, y_d=y[d], Sigma_y=Sigma_y, phi=phi)
                log_joint += log_joint_bA_d
                bA[d] = deepcopy(bA_d)

            # observation log joint
            for d in range(D):
                mean = np.dot(bA[d], phi)
                log_joint += multivariate_normal.logpdf(y[d], mean=mean, cov=Sigma_y)

            samples['y0'].append(deepcopy(y0))
            samples['LAMBDA'].append(deepcopy(LAMBDA))
            samples['b'].append(deepcopy(bA[:, 0]))
            samples['v'].append(deepcopy(v))
            samples['u'].append(deepcopy(u))
            samples['A'].append(deepcopy(bA[:, 1:]))
            log_joints.append(log_joint)

            print('Log joint: {}'.format(log_joint))
            print('------------------------------------------------')

        return samples, log_joints
    
    
    
    
class W_ARHMM(object):
    def __init__(self, D, Q, K, sigma2_y0=1e0, pi0 = None, alpha=5e0, alpha_diag_scale=4e1,
                 a1=5e1, sigma2_b=1e0, sigma2_A=1e-4, sigma2_A_diag=1e-4, A_diag = 1,
                 sigma2_y=1e0, a=1, c=1, RHO = -1, window=True):
        self.D = D
        self.Q = Q
        self.K = K
        self.Sigma_y0 = sigma2_y0 * np.eye(D)
        
        if pi0 == None:
            self.pi0 = np.ones(K) / K
        else:
            self.pi0 = pi0
            
        self.alpha = alpha
        
        self.alpha_diag_scale = alpha_diag_scale
        
        self.a1 = a1

        self.Sigma_b = sigma2_b * np.eye(D)

        self.Sigma_A = sigma2_A * np.eye(D)
        
        self.sigma2_A_diag = sigma2_A_diag
        
        self.A_diag = A_diag

        self.sigma2_y = sigma2_y
        
        self.C = (np.cos(a * np.log(1 + c)) + 1) / 2
        self.a = a
        self.c = c
        self.RHO = RHO
        
        self.window = window

        assert (Q < D)

        self._initialize()

    # initialize model params
    def _initialize(self):
        D, Q, K = self.D, self.Q, self.K
        Sigma_y0, a1, Sigma_b = self.Sigma_y0, self.a1, self.Sigma_b
        Sigma_A, sigma2_A_diag, A_diag = self.Sigma_A, self.sigma2_A_diag, self.A_diag
        pi0, alpha, alpha_diag_scale = self.pi0, self.alpha, self.alpha_diag_scale

        initial_log_joint = 0

        mean0_D = np.zeros(D)
        mean0_Q = np.zeros(Q)

        # Initialize latent variables: y0
        y0 = multivariate_normal.rvs(mean=mean0_D, cov=Sigma_y0)
        initial_log_joint += multivariate_normal.logpdf(y0, mean=mean0_D, cov=Sigma_y0)
        self.y0 = y0
        
        # Initialize latent variables: z0
        z0 = npr.choice(a=K, p=pi0)
        initial_log_joint += np.log(pi0[z0])
        self.z0 = z0

        P = np.zeros((K,K))
        LAMBDA = np.zeros(K)
        b = np.zeros((K,D))
        v = np.zeros((K,D,Q))
        u = np.zeros((K,D,Q))
        A = np.zeros((K,D,D))
        
        Sigma_A_diag = np.tile(Sigma_A[np.newaxis,:,:], (D,1,1))
        for d in range(D):
            Sigma_A_diag[d,d,d] = sigma2_A_diag
        
        for k in range(K):
            # Initialize latent variables: P
            alpha_K = np.ones(K) * alpha
            alpha_K[k] *= alpha_diag_scale
            P[k] = dirichlet.rvs(alpha = alpha_K)
            initial_log_joint += dirichlet.logpdf(P[k], alpha=alpha_K)
            
            # Initialize latent variables: LAMBDA
            LAMBDA[k] = invgamma.rvs(a=a1, scale=1)
            initial_log_joint += invgamma.logpdf(LAMBDA[k], a=a1, scale=1)

            # initialize latent variables: b
            b[k] = multivariate_normal.rvs(mean=mean0_D, cov=Sigma_b)
            initial_log_joint += multivariate_normal.logpdf(b[k], mean=mean0_D, cov=Sigma_b)

            # initialize latent variables: v,u
            v_k = np.zeros((D, Q))
            u_k = np.zeros((D, Q))
            
            # v_k, u_k covariance
            Sigma_vu = LAMBDA[k] * np.eye(Q)
            
            for d in range(D):
                v_kd = multivariate_normal.rvs(mean=mean0_Q, cov=Sigma_vu)
                initial_log_joint += multivariate_normal.logpdf(v_kd, mean=mean0_Q, cov=Sigma_vu)

                u_kd = multivariate_normal.rvs(mean=mean0_Q, cov=Sigma_vu)
                initial_log_joint += multivariate_normal.logpdf(u_kd, mean=mean0_Q, cov=Sigma_vu)

                v_k[d] = v_kd
                u_k[d] = u_kd

            v[k] = v_k
            u[k] = u_k
            
            # initialize latent variables: A
            A_k = np.zeros((D, D))
            for d in range(D):
                mean = np.dot(u[k,d], np.transpose(v[k]))
                mean[d] = A_diag
                A_kd = multivariate_normal.rvs(mean=mean, cov=Sigma_A_diag[d])
                initial_log_joint += multivariate_normal.logpdf(A_kd, mean=mean, cov=Sigma_A_diag[d])
                
                A_k[d] = A_kd

            A[k] = A_k

        self.P = P
        self.log_P = np.log(P)
        self.LAMBDA = LAMBDA
        self.b = b
        self.v = v
        self.u = u
        self.A = A

        self.initial_log_joint = initial_log_joint

    # samples states and observations
    def sample(self, T):
        D, K = self.D, self.K
        a, c, C, sigma2_y = self.a, self.c, self.C, self.sigma2_y
        z0, pi0, P, b, A = self.z0, self.pi0, self.P, self.b, self.A

        bA = np.concatenate((b[:,:, np.newaxis], A), axis=-1)

        y = np.zeros((D, T))
        z = -1 * np.ones(T).astype(int)
        
        # z_tt: z_t-1
        z_tt = z0

        for t in range(T):
            z[t] = npr.choice(a=K, p=P[z_tt])

            phi_t = self._compute_phi_t(y, t)
            mean = np.dot(bA[z[t]], phi_t)
            
            y[:, t] = multivariate_normal.rvs(mean=mean, cov=sigma2_y * np.eye(D))
            
            z_tt = z[t]

        return z, y, self.initial_log_joint

    def compute_initial_log_joint(self, y):
        D, K, T = self.D, self.K, y.shape[1]
        y0, pi0, P, b, A = self.y0, self.pi0, self.P, self.b, self.A
        Sigma_y1 = self.sigma2_y * np.eye(D)

        bA = np.concatenate((b[:,:, np.newaxis], A), axis=-1)
        phi = self._compute_phi(y0, y)
        
        log_likes = self._compute_log_likes(y=y, bA = bA, phi=phi)
        log_joint = self._hmm_normalizer(pi0, P, log_likes)

        return self.initial_log_joint + log_joint

    def _compute_phi_t(self, y, t):
        D = y.shape[0]
        T = y.shape[1]
        y0 = self.y0
        a, c, C = self.a, self.c, self.C
        window = self.window

        RHO = self.RHO

        phi_t = np.ones(D + 1)  # (D+1,T)
        

        if t == 0:
            if window:
                phi_t[1:] = deepcopy(C * y0)
            else:
                phi_t[1:] = deepcopy(y0)
            return phi_t
        else:
            if window:
                phi_t_temp = np.zeros(D)
                for tau in range(1, t + 1):
                    C_tau = a * np.log(tau + c)
                    if C_tau >= RHO - np.pi and C_tau <= RHO + np.pi:
                        C_tau_cos = (np.cos(C_tau - RHO) + 1) / 2
                    else:
                        C_tau_cos = 0

                    phi_t_temp += C_tau_cos * y[:, t - tau]

                phi_t[1:] = deepcopy(phi_t_temp)
            else:
                phi_t[1:] = deepcopy(y[:,t-1])

            return phi_t

    def _compute_phi(self, y0, y):
        D, T = self.D, y.shape[1]
        a, c, C = self.a, self.c, self.C
        RHO = self.RHO
        window = self.window

        phi = np.ones((D + 1, T))

        for t in range(T):
            if t == 0:
                if window:
                    phi[1:, 0] = deepcopy(C * y0)
                else:
                    phi[1:, 0] = deepcopy(y0)
                continue

            phi_t = np.zeros(D)
            
            if window:
                for tau in range(1, t + 1):
                    C_tau = a * np.log(tau + c)
                    if C_tau >= RHO - np.pi and C_tau <= RHO + np.pi:
                        C_tau_cos = (np.cos(C_tau - RHO) + 1) / 2
                    else:
                        C_tau_cos = 0
                    phi_t += C_tau_cos * y[:, t - tau]

                phi[1:, t] = deepcopy(phi_t)
            else:
                phi[1:, t] = deepcopy(y[:, t-1])

        return phi
    
    def initialize_z(self, K, T):
        
        z0, P = self.z0, self.P
        
        z = np.zeros((K,T))
        
        # z_tt: z_t-1
        z_tt = np.where(z0 == 1)[0][0]
        
        log_joint = 0

        for t in range(T):
            z[:, t] = multinomial.rvs(n=1, p=P[z_tt])
            log_joint += multinomial.logpmf(z[:, t], n=1, p=P[z_tt])
            z_tt = np.where(z[:, t] == 1)[0][0]
        
        return z, log_joint

    def _sample_y0(self, Sigma_y0, A, Sigma_y1, b, y1):
        window = self.window
        C = self.C
        Sigma_y0_inv = inv(Sigma_y0)
        Sigma_y1_inv = inv(Sigma_y1)
        A_t = np.transpose(A)
        
        if window:
            cov = inv(Sigma_y0_inv + C ** 2 * multi_dot([A_t, Sigma_y1_inv, A]))
            mean = C * (multi_dot([A_t, Sigma_y1_inv, y1]) - multi_dot([A_t, Sigma_y1_inv, b]))
        else:
            cov = inv(Sigma_y0_inv + multi_dot([A_t, Sigma_y1_inv, A]))
            mean = (multi_dot([A_t, Sigma_y1_inv, y1]) - multi_dot([A_t, Sigma_y1_inv, b]))
        mean = np.dot(cov, mean)  # np.transpose(np.dot(cov, mean))[0]

        y0 = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_y0 = multivariate_normal.logpdf(y0, mean=mean, cov=cov, allow_singular=True)

        return y0, log_joint_y0

    def _sample_v_d(self, A_d, Sigma_A, u, LAMBDA, Q):
        Sigma_v = LAMBDA * np.eye(Q)
        u_t = np.transpose(u)
        Sigma_A_inv = inv(Sigma_A)

        cov = inv(inv(Sigma_v) + multi_dot([u_t, Sigma_A_inv, u]))
        mean = np.dot(multi_dot([A_d, Sigma_A_inv, u]), cov)

        v_d = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_v_d = multivariate_normal.logpdf(v_d, mean=mean, cov=cov)

        return v_d, log_joint_v_d

    def _sample_u_d(self, A_d, Sigma_A, v, LAMBDA, Q):
        Sigma_u = LAMBDA * np.eye(Q)
        v_t = np.transpose(v)
        Sigma_A_inv = inv(Sigma_A)

        cov = inv(inv(Sigma_u) + multi_dot([v_t, Sigma_A_inv, v]))
        mean = np.dot(multi_dot([A_d, Sigma_A_inv, v]), cov)

        u_d = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_u_d = multivariate_normal.logpdf(u_d, mean=mean, cov=cov)

        return u_d, log_joint_u_d

    def _sample_bA_d(self, M, Sigma_bA, y_d, Sigma_y, phi):
        Sigma_bA_inv = inv(Sigma_bA)
        Sigma_y_inv = inv(Sigma_y)
        phi_t = np.transpose(phi)
        
        cov = inv(Sigma_bA_inv + multi_dot([phi, Sigma_y_inv, phi_t]))
        mean = np.dot(np.dot(M, Sigma_bA_inv) + multi_dot([y_d, Sigma_y_inv, phi_t]), cov)

        bA_d = multivariate_normal.rvs(mean=mean, cov=cov)
        log_joint_bA_d = multivariate_normal.logpdf(bA_d, mean=mean, cov=cov)

        return bA_d, log_joint_bA_d
    
    # modification of the ssm package: https://github.com/lindermanlab/ssm
    def _compute_mus(self, y, bA, phi):
        D, K, T = self.D, self.K, y.shape[1]
        
        mus = np.zeros((K,D,T))
        
        for k in range(K):
            mus[k] = np.dot(bA[k], phi)
        
        return mus
    
    # modification of the ssm package: https://github.com/lindermanlab/ssm
    def _compute_log_likes(self, y, bA, phi):
        K, D, T = self.K, self.D, y.shape[1]
        Sigma_y = self.sigma2_y * np.eye(D)
        
        mus = self._compute_mus(y, bA, phi)
        
        log_likes = np.zeros((K,T))
        
        for k in range(K):
            log_likes[k] = multivariate_normal_logpdf(y, mus[k], Sigma_y)
        
        return log_likes
    
    # from the SSM package: https://github.com/lindermanlab/ssm
    def _sample_z(self, pi0, Ps, log_likes):
        K, D, T = self.K, self.D, log_likes.shape[-1]
        
        # Forward pass gets the predicted state at time t given
        # observations up to and including those from time t
        alphas = np.zeros((K, T))
        self._forward_pass(pi0, Ps, log_likes, alphas)
        
        # Sample backward
        us = npr.rand(T)
        zs = -1 * np.ones(T)
        self._backward_sample(Ps, log_likes, alphas, us, zs)
        
        return zs
    
    # from the SSM package: https://github.com/lindermanlab/ssm
    def _backward_sample(self, Ps, log_likes, alphas, us, zs):
        T = log_likes.shape[1]
        K = log_likes.shape[0]

        lpzp1 = np.zeros(K)
        lpz = np.zeros(K)

        for t in range(T-1,-1,-1):
            # compute normalized log p(z[t] = k | z[t+1])
            lpz = lpzp1 + alphas[:,t]
            Z = logsumexp(lpz)

            # sample
            acc = 0
            zs[t] = K-1
            for k in range(K):
                acc += np.exp(lpz[k] - Z)
                if us[t] < acc:
                    zs[t] = k
                    break

            # set the transition potential
            if t > 0:
                lpzp1 = np.log(Ps[:, int(zs[t])] + LOG_EPS)
    
    # from the SSM package: https://github.com/lindermanlab/ssm
    def _forward_pass(self, pi0,
                 Ps,
                 log_likes,
                 alphas):

        T = log_likes.shape[1]  # number of time steps
        K = log_likes.shape[0]  # number of discrete states

        # Check if we have heterogeneous transition matrices.
        # If not, save memory by passing in log_Ps of shape (1, K, K)
        alphas[:,0] = np.log(pi0) + log_likes[:,0]
        for t in range(T-1):
            m = np.max(alphas[:,t])
            alphas[:,t+1] = np.log(np.dot(np.exp(alphas[:,t] - m), Ps)) + m + log_likes[:,t+1]
        return logsumexp(alphas[:,T-1])
    
    # from the SSM package: https://github.com/lindermanlab/ssm
    def _hmm_normalizer(self, pi0, Ps, ll):
        T, K = ll.shape[1], self.K
        alphas = np.zeros((K, T))
        
        # Make sure everything is C contiguous
        pi0 = to_c(pi0)
        Ps = to_c(Ps)
        ll = to_c(ll)

        self._forward_pass(pi0, Ps, ll, alphas)
        return logsumexp(alphas[-1])

    # run gibbs sampler
    def run_gibbs_sampler(self, y, iterations=100):
        D, Q, K, T = self.D, self.Q, self.K, y.shape[1]
        y0, z0, P, LAMBDA, b = deepcopy((self.y0, self.z0, self.P, self.LAMBDA, self.b))
        u, v, A = deepcopy((self.u, self.v, self.A))
        Sigma_y0, a1, Sigma_b = self.Sigma_y0, self.a1, self.Sigma_b
        Sigma_A, sigma2_A_diag, A_diag = self.Sigma_A, self.sigma2_A_diag, self.A_diag
        sigma2_y, Sigma_y1, Sigma_y = self.sigma2_y, self.sigma2_y * np.eye(D), self.sigma2_y * np.eye(T)
        pi0, alpha, alpha_diag_scale = self.pi0, self.alpha, self.alpha_diag_scale
        initial_log_joint = self.initial_log_joint
        C, window = self.C, self.window

        assert (D == y.shape[0])

        samples = defaultdict(list)
        log_joints = []
        
        # Concatenate z0 & z
        z0_T = -1 * np.ones(T+1).astype(int)
        
        # Compute PHI:
        phi = self._compute_phi(y0, y)
        
        # Concatenate b & A:
        bA = np.concatenate((b[:, :, np.newaxis], A), axis=-1)
        vec_Sigma_bA = np.zeros(D + 1)
        vec_Sigma_bA[0] = self.Sigma_b[0,0]
        vec_Sigma_bA[1:] = np.diag(Sigma_A)
        Sigma_bA = np.diag(vec_Sigma_bA)
        
        Sigma_bA_diag = np.tile(Sigma_bA[np.newaxis,:,:], (D,1,1))
        for d in range(D):
            Sigma_bA_diag[d,d+1,d+1] = sigma2_A_diag
            
        # indices for selected off-diagonal entries
        off_diag = np.zeros((D,D-1)).astype(int)
        idx_range = list(range(D))
        for d in range(D):
            off_diag[d] = idx_range[:d] + idx_range[d+1:]
        
        # Sample z0
        z0_T[0] = z0
        
        # Sample z
        log_likes = self._compute_log_likes(y=y, bA = bA, phi=phi)
        initial_log_joint += self._hmm_normalizer(pi0, P, log_likes)
        z0_T[1:] = self._sample_z(pi0, P, log_likes)
        
        # Initial samples
        samples['y0'].append(deepcopy(y0))
        samples['z0'].append(deepcopy(z0_T[0]))
        samples['P'].append(deepcopy(P))
        samples['LAMBDA'].append(deepcopy(LAMBDA))
        samples['b'].append(deepcopy(b))
        samples['v'].append(deepcopy(v))
        samples['u'].append(deepcopy(u))
        samples['A'].append(deepcopy(A))
        samples['z'].append(deepcopy(z0_T[1:]))

        log_joints.append(initial_log_joint)

        print('Initial log joint: {}'.format(initial_log_joint))
        print('------------------------------------------------')
        
        start = time.time()

        for itr in range(iterations):
            print('Iteration {}'.format(itr))
            log_joint = 0
            
            # SAMPLE y0
            z1 = z0_T[1]
            y0, log_joint_y0 = self._sample_y0(Sigma_y0=Sigma_y0, A=bA[z1, :, 1:], Sigma_y1=Sigma_y1, b=bA[z1, :, 0], y1=y[:, 0])
            log_joint += log_joint_y0

            # Update phi
            if window:
                phi[1:, 0] = C * y0
            else:
                phi[1:, 0] = y0
            
            # Compute y_k and update for alpha
            alpha_update = np.zeros((K,K))
            for t in range(T):
                z_tt = z0_T[t]
                z_t = z0_T[t+1]
                alpha_update[z_tt, z_t] += 1

            for k in range(K):
                # SAMPLE P
                new_alpha = np.ones(K) * alpha
                new_alpha[k] *= alpha_diag_scale
                new_alpha += alpha_update[k]
                P[k] = dirichlet.rvs(alpha=new_alpha)
                log_joint += dirichlet.logpdf(P[k], alpha=new_alpha)
                
                # SAMPLE LAMBDA
                new_a = a1 + D * Q
                new_scale = 1 + np.sum(np.add(np.square(v[k]), np.square(u[k]))) / 2
                LAMBDA[k] = invgamma.rvs(a=new_a, scale=new_scale)
                log_joint += invgamma.logpdf(LAMBDA[k], a=new_a, scale=new_scale)

                # SAMPLE v
                for d in range(D):
                    v_kd, log_joint_v_kd = self._sample_v_d(A_d=bA[k, off_diag[d], d+1], Sigma_A=Sigma_A[1:,1:], u=u[k, off_diag[d]], LAMBDA=LAMBDA[k], Q=Q)
                    log_joint += log_joint_v_kd
                    v[k,d] = v_kd

                # SAMPLE u
                for d in range(D):
                    u_kd, log_joint_u_kd = self._sample_u_d(A_d=bA[k, d, 1:][off_diag[d]], Sigma_A=Sigma_A[1:,1:], v=v[k, off_diag[d]], LAMBDA=LAMBDA[k], Q=Q)
                    log_joint += log_joint_u_kd
                    u[k,d] = u_kd

                # v_prime = v with a row padded to the top
                row = np.zeros((1, Q))
                v_prime = np.concatenate((row, v[k]), axis=0)
                
                k_idx = np.where(z0_T[1:] == k)[0]
                y_k = y[:, k_idx]
                phi_k = phi[:, k_idx]

                # SAMPLE bA
                for d in range(D):
                    M = np.dot(u[k,d], v_prime.T)
                    M[d+1] = A_diag
                    bA_kd, log_joint_bA_kd = self._sample_bA_d(M=M, Sigma_bA=Sigma_bA_diag[d], y_d=y_k[d],
                                                               Sigma_y=sigma2_y*np.eye(len(k_idx)), phi=phi_k)
                    log_joint += log_joint_bA_kd
                    bA[k, d] = deepcopy(bA_kd)
            
            # SAMPLE z0
            new_pi0 = lse(np.log(pi0) + np.log(P[:,z1]))
            z0_T[0] = npr.choice(a=K, p=new_pi0)
            log_joint += np.log(pi0[z0_T[0]])  
                    
            # SAMPLE z
            log_likes = self._compute_log_likes(y, bA = bA, phi=phi)
            z0_T[1:] = self._sample_z(new_pi0, P, log_likes)
            log_joint += self._hmm_normalizer(new_pi0, P, log_likes)

            samples['y0'].append(deepcopy(y0))
            samples['z0'].append(deepcopy(z0_T[0]))
            samples['P'].append(deepcopy(P))
            samples['LAMBDA'].append(deepcopy(LAMBDA))
            samples['b'].append(deepcopy(bA[:, :, 0]))
            samples['v'].append(deepcopy(v))
            samples['u'].append(deepcopy(u))
            samples['A'].append(deepcopy(bA[:, :, 1:]))
            samples['z'].append(deepcopy(z0_T[1:]))
            log_joints.append(log_joint)

            print('Log joint: {}'.format(log_joint))
            end = time.time()
            print("time elapsed: {}".format(end-start))
            start = end
            print('------------------------------------------------')

        return samples, log_joints    



# from the SSM package: https://github.com/lindermanlab/ssm
def flatten_to_dim(X, d):
    """
    Flatten an array of dimension k + d into an array of dimension 1 + d.

    Example:
        X = npr.rand(10, 5, 2, 2)
        flatten_to_dim(X, 4).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 3).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 2).shape # (50, 2, 2)
        flatten_to_dim(X, 1).shape # (100, 2)

    Parameters
    ----------
    X : array_like
        The array to be flattened.  Must be at least d dimensional

    d : int (> 0)
        The number of dimensions to retain.  All leading dimensions are flattened.

    Returns
    -------
    flat_X : array_like
        The input X flattened into an array dimension d (if X.ndim == d)
        or d+1 (if X.ndim > d)
    """
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])

# from the SSM package: https://github.com/lindermanlab/ssm
to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)
    
# from the SSM package: https://github.com/lindermanlab/ssm
def batch_mahalanobis(L, x):
    """
    Compute the squared Mahalanobis distance.
    :math:`x^T M^{-1} x` for a factored :math:`M = LL^T`.

    Copied from PyTorch torch.distributions.multivariate_normal.

    Parameters
    ----------
    L : array_like (..., D, D)
        Cholesky factorization(s) of covariance matrix

    x : array_like (..., D)
        Points at which to evaluate the quadratic term

    Returns
    -------
    y : array_like (...,)
        squared Mahalanobis distance :math:`x^T (LL^T)^{-1} x`

        x^T (LL^T)^{-1} x = x^T L^{-T} L^{-1} x
    """
    # The most common shapes are x: (T, D) and L : (D, D)
    # Special case that one
    if x.ndim == 2 and L.ndim == 2:
        xs = solve_triangular(L, x, lower=True)
        return np.sum(xs**2, axis=0)

    # Flatten the Cholesky into a (-1, D, D) array
    flat_L = flatten_to_dim(L, 2)
    # Invert each of the K arrays and reshape like L
    L_inv = np.reshape(np.array([np.linalg.inv(Li.T) for Li in flat_L]), L.shape)
    # dot with L_inv^T; square and sum.
    xs = np.einsum('...i,...ij->...j', x, L_inv)
    return np.sum(xs**2, axis=-1)

# from the SSM package: https://github.com/lindermanlab/ssm
def multivariate_normal_logpdf(y, mus, Sigmas):
    """
    Compute the log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)

    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = y.shape[0]
    Ls = np.linalg.cholesky(Sigmas)                                  # (..., D, D)

    # Quadratic term
    lp = -0.5 * batch_mahalanobis(Ls, y - mus)                    # (...,)
    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det             # (...,)

    return lp    
    
# from the SSM package: https://github.com/lindermanlab/ssm
def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap

# from the SSM package: https://github.com/lindermanlab/ssm
def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm 
