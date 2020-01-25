import numpy as np
import nengo

def _make_delay_network(q, theta):
    Q = np.arange(q, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i - j + 1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B


def _make_nef_lti(tau, A, B):
    AH = tau * A + np.eye(A.shape[0])
    BH = tau * B
    return AH, BH


class GranuleGolgiCircuit(nengo.Network):
    def __init__(self,
                 ens_pc,
                 n_golgi=20,
                 n_granule=200,
                 n_eval_points=1000,
                 n_golgi_convergence=None,
                 n_granule_convergence=None,
                 q=6,
                 theta=0.4,
                 tau=60e-3,
                 solver_relax=False,
                 solver_reg=1e-4,
                 solver_tol=1e-2,
                 solver_renormalise=False,
                 golgi_max_rates=None,
                 granule_max_rates=None,
                 golgi_intercepts=None,
                 granule_intercepts=None,
                 use_nengo_bio=True):
        # Use nengo bio if the corresponding flag is set
        if use_nengo_bio:
            import nengo_bio as bio
            Ensemble = bio.Ensemble
            Connection = bio.Connection
        else:
            Ensemble = nengo.Ensemble
            Connection = nengo.Connection

        # Use biologically plausible maximum rates
        if golgi_max_rates is None:
            golgi_max_rates = nengo.dists.Uniform(50, 100)
        if granule_max_rates is None:
            granule_max_rates = nengo.dists.Uniform(50, 200)

        # Use cosine similarity for intercepts
        if golgi_intercepts is None:
            golgi_intercepts = nengo.dists.CosineSimilarity(q + 2)
        if granule_intercepts is None:
            granule_intercepts = nengo.dists.CosineSimilarity(q + 2)

        # Copy the given parameters
        self.ens_pc = ens_pc
        self.n_golgi = n_golgi
        self.n_granule = n_granule
        self.n_eval_points = n_eval_points
        self.n_golgi_convergence = n_golgi_convergence
        self.n_granule_convergence = n_granule_convergence
        self.q = q
        self.theta = theta
        self.tau = tau
        self.solver_relax = solver_relax
        self.solver_reg = solver_reg
        self.solver_tol = solver_tol
        self.solver_renormalise = solver_renormalise
        self.golgi_max_rates = golgi_max_rates
        self.granule_max_rates = granule_max_rates
        self.golgi_intercepts = golgi_intercepts
        self.granule_intercepts = granule_intercepts
        self.use_nengo_bio = use_nengo_bio

        # Call the inherited network constructor
        super().__init__(label="Granule/Golgi Layer")

        # Build the golgi cell ensemble
        with self:
            kwargs_golgi = {
                'n_neurons': self.n_golgi,
                'dimensions': self.q,
                'intercepts': self.golgi_intercepts,
                'max_rates': self.golgi_max_rates,
            }
            if use_nengo_bio:
                kwargs_golgi['p_inh'] = 1.0
            self.ens_golgi = Ensemble(**kwargs_golgi)

            # Build the granule cell ensemble
            kwargs_granule = {
                'n_neurons': self.n_granule,
                'dimensions': self.q,
                'intercepts': self.granule_intercepts,
                'max_rates': self.granule_max_rates,
            }
            if use_nengo_bio:
                kwargs_granule['p_exc'] = 1.0
            self.ens_granule = Ensemble(**kwargs_granule)

            # Compute the Delay Network coefficients
            AH, BH = _make_nef_lti(
                tau, *_make_delay_network(q=self.q, theta=self.theta))

            # Make the recurrent connections
            if use_nengo_bio:
                # Assemble the argments that are being passed to the solver
                kwargs_solver = {
                    'relax': self.solver_relax,
                    'reg': self.solver_reg,
                    'extra_args': {
                        'tol': self.solver_tol,
                        'renormalise': self.solver_renormalise,
                    }
                }

                # Granule -> Golgi
                Connection((self.ens_pc, self.ens_granule, self.ens_golgi),
                           self.ens_golgi,
                           transform=np.concatenate((
                               BH.reshape(-1, 1),
                               AH,
                               np.zeros((AH.shape[0], AH.shape[0])),
                           ),
                                                   axis=1),
                           synapse_exc=self.tau,
                           synapse_inh=self.tau,
                           n_eval_points=self.n_eval_points,
                           max_n_post_synapses=self.n_golgi_convergence,
                           solver=bio.solvers.QPSolver(**kwargs_solver))

                # Golgi -> Granule
                Connection((self.ens_pc, self.ens_golgi),
                           self.ens_granule,
                           transform=np.concatenate((
                               BH.reshape(-1, 1),
                               AH,
                           ),
                                                   axis=1),
                           synapse_exc=self.tau,
                           synapse_inh=self.tau,
                           n_eval_points=self.n_eval_points,
                           max_n_post_synapses=self.n_granule_convergence,
                           solver=bio.solvers.QPSolver(**kwargs_solver))
            else:
                # Input -> Granule
                Connection(self.ens_pc,
                           self.ens_granule,
                           transform=BH,
                           synapse=tau)

                # Input -> Golgi
                Connection(self.ens_pc,
                           self.ens_golgi,
                           transform=BH,
                           synapse=tau)

                # Granule -> Golgi
                Connection(self.ens_granule,
                           self.ens_golgi,
                           transform=AH,
                           synapse=tau)

                # Golgi -> Granule
                Connection(self.ens_golgi,
                           self.ens_granule,
                           transform=AH,
                           synapse=tau)

