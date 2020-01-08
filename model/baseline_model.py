import nengo
import nengolib
import numpy as np

from eyeblink_motor_system import EyeblinkReflex, Eyelid

def make_lmu(q=6, theta=1.0):
    # Do Aaron's math to generate the matrices
    #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
    Q = np.arange(q, dtype=np.float64)
    R = (2*Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B

def puff_func(t):
    if 3.0 > t % 3 > 2.9:
        return 1
    else:
        return 0

def tone_func(t):
    if 2.8 > t % 3 > 2.7:
        return 1
    else:
        return 0

model = nengo.Network()
with model:
    ###########################################################################
    # Setup the conditioned stimulus (i.e., a tone) and the unconditioned     #
    # stimulus (i.e., a puff)                                                 #
    ###########################################################################
    nd_tone = nengo.Node(tone_func)
    nd_puff = nengo.Node(puff_func)


    ###########################################################################
    # Setup the reflex generator and the eye motor system                     #
    ###########################################################################

    # Scaling factor that has to be applied to the reflex trajectory to scale
    # it to a range from 0 to 1
    reflex_scale = 1.0 / 25.0

    # The reflex system takes an input and produces the reflex trajectory on
    # the rising edge (convolves the differential of the input with the
    # trajectory)
    nd_reflex = nengo.Node(EyeblinkReflex()) # Unscaled output
    nd_reflex_out = nengo.Node(size_in=1)    # Normalised output
    nengo.Connection(nd_reflex[0], nd_reflex_out, transform=reflex_scale,
                     synapse=None)

    # The eyelid component represents the state of the eye in the world.
    # It receives two inputs, an agonist input (closing the eye, dim 0) and an
    # antagonist input (opening the eye, dim 1).
    nd_eyelid = nengo.Node(Eyelid())         # Unscaled input
    nd_eyelid_in = nengo.Node(size_in=1)     # Normalised input
    nengo.Connection(nd_eyelid_in, nd_eyelid[0], transform=1.0 / reflex_scale,
                     synapse=None)

    # Constantly open the eye a little bit
    nd_eye_bias = nengo.Node(lambda _: 0.5)
    nengo.Connection(nd_eye_bias, nd_eyelid[1])

    # We can't detect the puff if the eye is closed, multiply the output from
    # nd_puff with the amount the eye is opened. This is our unconditioned
    # stimulus
    c0, c1 = nengo.Node(size_in=1), nengo.Node(size_in=1) # Only for GUI
    nd_us = nengo.Node(lambda t, x: x[0] * (1 - x[1]), size_in=2, size_out=1)
    nengo.Connection(nd_puff, nd_us[0], synapse=None)
    nengo.Connection(nd_eyelid, c0, synapse=None)
    nengo.Connection(c0, c1, synapse=None)
    nengo.Connection(c1, nd_us[1], synapse=None)

    # Connect the unconditioned stimulus to the reflex generator
    nengo.Connection(nd_us, nd_reflex)
    nengo.Connection(nd_reflex_out, nd_eyelid_in)

    ###########################################################################
    # Generate a neural representation of the conditioned stimulus            #
    ###########################################################################

    nd_cs = nengo.Node(size_in=1)
    ens_pcn = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(nd_tone, nd_cs, synapse=None)
    nengo.Connection(nd_cs, ens_pcn)

    ###########################################################################
    # Generate a LMU representation of the conditioned stimulus               #
    ###########################################################################

    # Recurrent connection time constant
    tau = 1.0

    # Create the LTI and transform it into the corresponding NEF LTI
    A, B = make_lmu(q = 12, theta = 0.3)
    Ap = A + tau * np.eye(A.shape[0])
    Bp = B * tau

    # Build the LMU, feed the conditioned stimulus into it
    # TODO: Replace with nengo-bio LMU
    ens_granule = nengo.Ensemble(n_neurons=500, dimensions=A.shape[0])
    nengo.Connection(ens_granule, ens_granule, transform=Ap, synapse=tau)
    nengo.Connection(ens_pcn, ens_granule, transform=Bp, synapse=tau)

    ###########################################################################
    # Learn the connection from the Granule cells to the Purkinje cells via   #
    # input from the Interior Olive                                           #
    ###########################################################################

    ens_cn = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_io = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_purkinje = nengo.Ensemble(n_neurons=100, dimensions=1)

    # Represent the error signal in ens_io
    nengo.Connection(nd_reflex_out[0], ens_io, transform=-1)
    nengo.Connection(ens_cn, ens_io, transform=1) # This connection does not exist

    # Project from the 
    c_learn = nengo.Connection(
        ens_granule.neurons, ens_purkinje.neurons,
        transform=np.zeros((ens_purkinje.n_neurons, ens_granule.n_neurons)),
        learning_rule_type=nengo.learning_rules.PES(learning_rate=3e-5))
    nengo.Connection(ens_io, c_learn.learning_rule)

    ###########################################################################
    # Project from CN onto the motor system
    ###########################################################################

    nengo.Connection(ens_purkinje, ens_cn)
    nengo.Connection(ens_cn, nd_eyelid_in[0])
