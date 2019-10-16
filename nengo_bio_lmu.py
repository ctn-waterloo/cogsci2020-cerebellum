import nengo
import nengo_bio as bio
import numpy as np

bio.set_defaults()

def make_delay_network(q, theta):
    Q = np.arange(q, dtype=np.float64)
    R = (2*Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B

q = 3
theta = 0.5
A, B = make_delay_network(q, theta)

with nengo.Network() as model:
    stim = nengo.Node(size_in=1)
    ens_stim = bio.Ensemble(n_neurons=100, dimensions=1, p_exc=1)
    ens_golgi = bio.Ensemble(n_neurons=100, dimensions=A.shape[0], p_inh=1)
    ens_granule = bio.Ensemble(n_neurons=1000, dimensions=A.shape[0], p_exc=1)
    ens_bias = bio.Ensemble(n_neurons=100, dimensions=1, p_inh=1,
        encoders=np.ones((100, 1)),
        intercepts=nengo.dists.Uniform(-0.9, -0.1),
        eval_points=nengo.dists.Choice([np.zeros(1)]))

    tau=0.1
    AH = A * tau + np.eye(A.shape[0])
    BH = B * tau
    nengo.Connection(stim, ens_stim)

    bio.Connection((ens_stim, ens_granule, ens_bias), ens_golgi,
        transform=np.concatenate((
            BH.reshape(-1, 1),
            AH,
            np.zeros((AH.shape[0], 1))
        ), axis=1),
        synapse_exc=tau,
        synapse_inh=tau,
        max_n_post_synapses=100,
        solver=bio.solvers.QPSolver(relax=False, reg=1e-3)
    )
    bio.Connection((ens_stim, ens_golgi), ens_granule,
        transform=np.concatenate((
            BH.reshape(-1, 1),
            AH,
        ), axis=1),
        synapse_exc=tau,
        synapse_inh=tau,
        max_n_post_synapses=50,
        solver=bio.solvers.QPSolver(relax=False, reg=1e-3)
    )
