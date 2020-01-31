import numpy as np
import nengo


def _make_delay_network(q, theta):
    Q = np.arange(q, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i - j + 1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B


def _discretize_lti(dt, A, B):
    import scipy.linalg
    Ad = scipy.linalg.expm(A * dt)
    Bd = np.dot(np.dot(np.linalg.inv(A), (Ad - np.eye(A.shape[0]))), B)
    return Ad, Bd


def _make_nef_lti(tau, A, B):
    AH = tau * A + np.eye(A.shape[0])
    BH = tau * B
    return AH, BH


class Legendre(nengo.Process):
    def __init__(self, theta, q):
        self.q = q
        self.theta = theta
        self.A, self.B = _make_delay_network(self.q, self.theta)
        super().__init__(default_size_in=1, default_size_out=q)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state = np.zeros(self.q)
        Ad, Bd = _discretize_lti(dt, self.A, self.B)

        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x)
            return state

        return step_legendre


class GranuleGolgiCircuit(nengo.Network):
    def _kwargs_golgi(self):
        return {
            'n_neurons': self.n_golgi,
            'dimensions': self.q,
            'intercepts': self.golgi_intercepts,
            'max_rates': self.golgi_max_rates,
        }

    def _kwargs_granule(self):
        return {
            'n_neurons': self.n_granule,
            'dimensions': self.q,
            'intercepts': self.granule_intercepts,
            'max_rates': self.granule_max_rates,
        }

    def _build_direct_mode_network(self):
        with self:
            # Create a direct implementation of the delay network
            self.nd_delay_network = nengo.Node(Legendre(self.theta, self.q))

            # Create a "ens_granule" ensemble as the intermediate layer we're
            # learning from
            kwargs_granule = self._kwargs_granule()
            self.ens_granule = nengo.Ensemble(**kwargs_granule)

            # Connect the network up
            nengo.Connection(self.ens_pc, self.nd_delay_network)
            nengo.Connection(self.nd_delay_network, self.ens_granule)


    def _build_single_population_network(self):
        use_esn = self.mode == "echo_state"
        with self:
            # Create the "granule" ensemble
            kwargs_granule = self._kwargs_granule()
            self.ens_granule = nengo.Ensemble(**kwargs_granule)

            if not use_esn:
                # Compute the Delay Network coefficients
                AH, BH = _make_nef_lti(
                    self.tau, *_make_delay_network(q=self.q, theta=self.theta))
            else:
                # Do not alter the random state
                rs = np.random.get_state()
                try:
                    import scipy.sparse
                    # Adapted from "generate_internal_weights.m" in
                    # http://minds.jacobs-university.de/uploads/papers/freqGen.zip
                    A = scipy.sparse.random(
                        self.q,
                        self.q,
                        min(5.0 / self.q, 1)
                    ).toarray()
                    A[A != 0] = A[A != 0] - 0.5
                    maxEigVal = maxVal = np.max(np.abs(
                        np.linalg.eigvals(A)));
                    A = A / (maxEigVal)

                    B = np.random.uniform(-1, 1, (
                        self.q,
                        1
                    ))

                    AH, BH = _make_nef_lti(self.tau, 25 * A, B)
                finally:
                    np.random.set_state(rs)


            # Build the recurrent connection
            nengo.Connection(
                self.ens_granule,
                self.ens_granule,
                transform=AH,
                synapse=self.tau
            )

            # Build the input connection
            nengo.Connection(
                self.ens_pc,
                self.ens_granule,
                transform=BH,
                synapse=self.tau,
            )


    def _build_two_population_network(self):
        # Decide whether or not nengo bio should be used
        use_nengo_bio = self.mode == "two_populations_dales_principle"
        if use_nengo_bio:
            import nengo_bio as bio
            Ensemble = bio.Ensemble
            Connection = bio.Connection
        else:
            Ensemble = nengo.Ensemble
            Connection = nengo.Connection

        with self:
            # Build the golgi cell ensemble
            kwargs_golgi = self._kwargs_golgi()
            if use_nengo_bio:
                kwargs_golgi['p_inh'] = 1.0
            self.ens_golgi = Ensemble(**kwargs_golgi)

            # Build the granule cell ensemble
            kwargs_granule = self._kwargs_granule()
            if use_nengo_bio:
                kwargs_granule['p_exc'] = 1.0
            self.ens_granule = Ensemble(**kwargs_granule)

            # Compute the Delay Network coefficients
            AH, BH = _make_nef_lti(
                self.tau, *_make_delay_network(q=self.q, theta=self.theta))

            self.nd_state = nengo.Node(size_in=self.q)

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
                           synapse=self.tau)

                # Input -> Golgi
                Connection(self.ens_pc,
                           self.ens_golgi,
                           transform=BH,
                           synapse=self.tau)

                # Granule -> Golgi
                Connection(self.ens_granule,
                           self.ens_golgi,
                           transform=AH,
                           synapse=self.tau)

                # Golgi -> Granule
                Connection(self.ens_golgi,
                           self.ens_granule,
                           transform=AH,
                           synapse=self.tau)

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
                 mode="two_populations"):

        # Make sure the give mode is valid
        valid_modes = {
            "direct",
            "echo_state",
            "single_population",
            "two_populations",
            "two_populations_dales_principle",
        }
        if not mode in valid_modes:
            raise ValueError("\"mode\" must be one of {}".format(
                str(valid_modes)))

        # Determine whether or not to use nengo bio
        use_nengo_bio = mode == "two_populations_dales_principle"

        # Use biologically plausible maximum rates
        if golgi_max_rates is None:
            golgi_max_rates = nengo.dists.Uniform(50, 100)
        if granule_max_rates is None:
            granule_max_rates = nengo.dists.Uniform(50, 100)

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
        self.mode = mode

        # Call the inherited network constructor
        super().__init__(label="Granule/Golgi Layer")

        # Instantiate different circuits depending on the mode we're using
        if mode in {
                "direct",
        }:
            self._build_direct_mode_network()
        elif mode in {"echo_state", "single_population"}:
            self._build_single_population_network()
        elif mode in {"two_populations", "two_populations_dales_principle"}:
            self._build_two_population_network()
        else:
            assert False

