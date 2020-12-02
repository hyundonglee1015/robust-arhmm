import numpy as np
from scipy.stats import norm, invgamma, multivariate_normal, multinomial, dirichlet
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from copy import deepcopy
from numpy.linalg import inv, multi_dot, matrix_rank
import matplotlib.pyplot as plt
import numpy.random as npr
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
class ARHMM(object):
    def __init__(self, D, Q, K, sigma2_y0=1e0, pi_0 = None, alpha = 1e1, P_diag = 1e1,
                 a1=5e1, sigma2_b=1e0, sigma2_A=1e-4,
                 sigma2_y=1e0, a=1, c=1, RHO = -1):
        self.D = D
        self.Q = Q
        self.K = K
        self.Sigma_y0 = sigma2_y0 * np.eye(D)
        
        if pi_0 == None:
            self.pi_0 = np.ones(K) / K
        else:
            self.pi_0 = pi_0
            
        self.alpha = alpha
        
        self.P_diag = P_diag
        
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
        D, Q, K = self.D, self.Q, self.K
        Sigma_y0, a1, Sigma_b, Sigma_A = self.Sigma_y0, self.a1, self.Sigma_b, self.Sigma_A
        pi_0, alpha, P_diag = self.pi_0, self.alpha, self.P_diag

        initial_log_joint = 0

        mean0_D = np.zeros(D)
        mean0_Q = np.zeros(Q)

        # Initialize latent variables: y0
        y0 = multivariate_normal.rvs(mean=mean0_D, cov=Sigma_y0)
        initial_log_joint += multivariate_normal.logpdf(y0, mean=mean0_D, cov=Sigma_y0)
        self.y0 = y0
        
        # Initialize latent variables: z0
        z0 = multinomial.rvs(n=1, p=pi_0)
        initial_log_joint += multinomial.logpmf(z0, n=1, p=pi_0)
        self.z0 = z0

        P = np.zeros((K,K))
        LAMBDA = np.zeros(K)
        b = np.zeros((K,D))
        v = np.zeros((K,D,Q))
        u = np.zeros((K,D,Q))
        A = np.zeros((K,D,D))
        for k in range(K):
            # Initialize latent variables: P
            alpha_K = np.ones(K) * alpha
            alpha_K[k] = alpha * P_diag
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
            for d in range(self.D):
                Sigma_vu = LAMBDA[k] * np.eye(Q)
                v_kd = multivariate_normal.rvs(mean=mean0_Q, cov=Sigma_vu)
                initial_log_joint += multivariate_normal.logpdf(v_kd, mean=mean0_Q, cov=Sigma_vu)

                u_kd = multivariate_normal.rvs(mean=mean0_Q, cov=Sigma_vu)
                initial_log_joint += multivariate_normal.logpdf(u_kd, mean=mean0_Q, cov=Sigma_vu)

                # should i deepcopy?
                v_k[d] = v_kd
                u_k[d] = u_kd

            v[k] = deepcopy(v_k)
            u[k] = deepcopy(u_k)

            # initialize latent variables: A
            A_k = np.zeros((D, D))
            for d in range(D):
                mean = np.dot(u[k,d], np.transpose(v[k]))
                A_kd = multivariate_normal.rvs(mean=mean, cov=Sigma_A)
                initial_log_joint += multivariate_normal.logpdf(A_kd, mean=mean, cov=Sigma_A)

                # should I deepcopy?
                A_k[d] = A_kd
                
            A[k] = deepcopy(A_k)

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
        z0, P, b, A = self.z0, self.P, self.b, self.A

        bA = np.concatenate((b[:,:, np.newaxis], A), axis=-1)

        y = np.zeros((D, T))
        z = np.zeros((K, T))
        log_joint = 0
        
        # z_tt: z_t-1
        z_tt = np.where(z0 == 1)[0][0]
        
        for t in range(T):
            z[:, t] = multinomial.rvs(n=1, p=P[z_tt])
            log_joint += multinomial.logpmf(z[:, t], n=1, p=P[z_tt])
            z_t = np.where(z[:, t] == 1)[0][0]
            
            phi_t = self.compute_phi_t(y, t)
            mean = np.dot(bA[z_t], phi_t)
            y[:, t] = multivariate_normal.rvs(mean=mean, cov=sigma2_y * np.eye(D))
            log_joint += multivariate_normal.logpdf(y[:, t], mean=mean, cov=sigma2_y * np.eye(D))
            
            z_tt = z_t

        z_transpose = np.transpose(np.where(z == 1))
        true_states = np.transpose(sorted(z_transpose, key=lambda z_tp: z_tp[1]))[0]

        return true_states, y, self.initial_log_joint + log_joint

    def compute_initial_log_joint(self, y, z, phi):
        D, K, T = self.D, self.K, y.shape[1]
        y0, b, A = self.y0, self.b, self.A
        Sigma_y1 = self.sigma2_y * np.eye(D)

        bA = np.concatenate((b[:,:, np.newaxis], A), axis=-1)

        log_joint = 0
        for t in range(T):
            k = np.where(z[:,t] == 1)[0][0]
            mean = np.dot(bA[k], phi[:,t])
            log_joint += multivariate_normal.logpdf(y[:,t], mean=mean, cov=Sigma_y1)

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
        D, Q, K, T = self.D, self.Q, self.K, y.shape[1]
        y0, z0, P, LAMBDA, b = deepcopy(self.y0), deepcopy(self.z0), deepcopy(self.P), deepcopy(self.LAMBDA), deepcopy(self.b)
        u, v, A = deepcopy(self.u), deepcopy(self.v), deepcopy(self.A)
        Sigma_y0, a1, Sigma_b, Sigma_A = self.Sigma_y0, self.a1, self.Sigma_b, self.Sigma_A
        sigma2_y, Sigma_y1, Sigma_y = self.sigma2_y, self.sigma2_y * np.eye(D), self.sigma2_y * np.eye(T)
        pi_0, alpha, P_diag = self.pi_0, self.alpha, self.P_diag
        C = self.C

        assert (D == y.shape[0])

        samples = defaultdict(list)
        log_joints = []
        
        z, z_log_joint = self.initialize_z(K, T)

        samples['y0'].append(deepcopy(y0))
        samples['z0'].append(deepcopy(z0))
        samples['P'].append(deepcopy(P))
        samples['LAMBDA'].append(deepcopy(LAMBDA))
        samples['b'].append(deepcopy(b))
        samples['v'].append(deepcopy(v))
        samples['u'].append(deepcopy(u))
        samples['A'].append(deepcopy(A))
        samples['z'].append(deepcopy(z))

        # Compute PHI:
        phi = self.compute_phi(y0, y)

        log_joint = self.compute_initial_log_joint(y, z, phi)
        
        log_joint += z_log_joint

        log_joints.append(log_joint)

        print('Initial log joint: {}'.format(log_joint))
        print('------------------------------------------------')

        # Concatenate b & A:
        bA = np.concatenate((b[:, :, np.newaxis], A), axis=-1)
        vec_Sigma_bA = np.zeros(D + 1)
        vec_Sigma_bA[0] = self.Sigma_b[0, 0]
        vec_Sigma_bA[1:] = np.diag(Sigma_A)
        Sigma_bA = np.diag(vec_Sigma_bA)
        
        # Concatenate z0 & z
        z0_T = np.concatenate((z0[:, np.newaxis], z), axis=-1)

        for itr in range(iterations):
            print('Iteration {}'.format(itr))
            log_joint = 0
            
            # SAMPLE y0
            z1 = np.where(z0_T[:, 1] == 1)[0][0]
            y0, log_joint_y0 = self._sample_y0(Sigma_y0=Sigma_y0,
                                               C=C, A=bA[z1, :, 1:], Sigma_y1=Sigma_y1, b=bA[z1, :, 0], y1=y[:, 0])
            log_joint += log_joint_y0

            # Update phi
            phi[1:, 0] = deepcopy(C * y0)
            
            # Compute y_k and update for alpha
            alpha_update = np.zeros((K,K))
            z_k = defaultdict(list)
            for t in range(T):
                z_tt = np.where(z0_T[:,t] == 1)[0][0]
                z_t = np.where(z0_T[:,t+1] == 1)[0][0]
                alpha_update[z_tt, z_t] += 1
                z_k[z_t].append(t)

            for k in range(K):
                # SAMPLE P
                new_alpha = np.ones(K) * alpha
                new_alpha[k] *= P_diag
                new_alpha += alpha_update[k]
                P[k] = dirichlet.rvs(alpha = new_alpha)
                log_joint += dirichlet.logpdf(P[k], alpha=new_alpha)
                
                # SAMPLE LAMBDA
                new_a = a1 + D * Q
                new_scale = 1 + np.sum(np.add(np.square(v[k]), np.square(u[k]))) / 2
                LAMBDA[k] = invgamma.rvs(a=new_a, scale=new_scale)
                log_joint += invgamma.logpdf(LAMBDA[k], a=new_a, scale=new_scale)

                # SAMPLE v
                for d in range(D):
                    v_d, log_joint_v_d = self._sample_v_d(A_d=bA[k, :, d+1], Sigma_A=Sigma_A, u=u[k], LAMBDA=LAMBDA[k], Q=Q)
                    log_joint += log_joint_v_d
                    v[k,d] = deepcopy(v_d)

                # SAMPLE u
                for d in range(D):
                    u_d, log_joint_u_d = self._sample_u_d(A_d=bA[k, d, 1:], Sigma_A=Sigma_A, v=v[k], LAMBDA=LAMBDA[k], Q=Q)
                    log_joint += log_joint_u_d
                    u[k,d] = deepcopy(u_d)

                # v_prime = v with a row padded to the top
                row = np.zeros((1, Q))
                v_prime = np.concatenate((row, v[k]), axis=0)
                
                y_k = y[:, z_k[k]]
                phi_k = phi[:, z_k[k]]

                # SAMPLE bA
                for d in range(D):
                    bA_d, log_joint_bA_d = self._sample_bA_d(M=np.dot(u[k,d], v_prime.T),
                                                             Sigma_bA=Sigma_bA, y_d=y_k[d], Sigma_y=sigma2_y*np.eye(len(z_k[k])), phi=phi_k)
                    log_joint += log_joint_bA_d
                    bA[k, d] = deepcopy(bA_d)
                    
            # SAMPLE z0
            new_pi0 = pi_0 * P[:,z1]
            new_pi0 = lse(np.log(new_pi0))
            #new_pi0 = lse(new_pi0)
            while np.sum(new_pi0) > 1:
                new_pi0 = new_pi0 / np.sum(new_pi0)
            z0_T[:, 0] = multinomial.rvs(n=1, p=new_pi0)
            log_joint += multinomial.logpmf(z0_T[:, 0], n=1, p=new_pi0)    
                    
            # SAMPLE z
            for t in range(T):
                z_tt = np.where(z0_T[:,t] == 1)[0][0]
                z_t = np.where(z0_T[:,t+1] == 1)[0][0]
                P_tt_update = deepcopy(P[z_tt])
                for k in range(K):
                    mean_k = np.dot(bA[k], phi[:, t])
                    P_tt_update[k] *= multivariate_normal.pdf(y[:, t], mean=mean_k, cov=Sigma_y1)

                P_tt_update = lse(np.log(P_tt_update))
                #P_tt_update = lse(P_tt_update)
                while np.sum(P_tt_update) > 1:
                    P_tt_update = P_tt_update / np.sum(P_tt_update)
                z0_T[:, t+1] = multinomial.rvs(n=1, p=P_tt_update)
                log_joint += multinomial.logpmf(z0_T[:, t+1], n=1, p=P_tt_update)

            # Compute observation log joint
            for t in range(T):
                z_t = np.where(z0_T[:,t+1] == 1)[0][0]
                mean = np.dot(bA[z_t], phi[:, t])
                log_joint += multivariate_normal.logpdf(y[:, t], mean=mean, cov=Sigma_y1)
                

            samples['y0'].append(deepcopy(y0))
            samples['z0'].append(deepcopy(z0_T[:, 0]))
            samples['P'].append(deepcopy(P))
            samples['LAMBDA'].append(deepcopy(LAMBDA))
            samples['b'].append(deepcopy(bA[:, :, 0]))
            samples['v'].append(deepcopy(v))
            samples['u'].append(deepcopy(u))
            samples['A'].append(deepcopy(bA[:, :, 1:]))
            samples['z'].append(deepcopy(z0_T[:, 1:]))
            log_joints.append(log_joint)

            print('Log joint: {}'.format(log_joint))
            print('------------------------------------------------')

        return samples, log_joints

    
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