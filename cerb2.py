import nengo
import nengolib
import numpy as np

model = nengo.Network()
with model:

    
    def stim_func(t):
        if 0.8 > t%1 > 0.7:
            return 1
        else:
            return 0
    stim = nengo.Node(stim_func)
    
    def puff_func(t):
        if 1.0 > t%1 > 0.9:
            return 1
        else:
            return 0
        
    puff = nengo.Node(puff_func)


    
    process = nengo.processes.WhiteSignal(10.0, high=15, y0=0)
    neuron_type = nengo.LIF()

    rw = nengolib.networks.RollingWindow(
        theta=0.5, n_neurons=500, process=process, neuron_type=neuron_type)

    nengo.Connection(stim, rw.input, synapse=None)    
    
    
    purkinje_pause = nengo.Ensemble(n_neurons=100, dimensions=1,
                                    encoders=nengo.dists.Choice([[-1]]),
                                    intercepts=nengo.dists.Uniform(-2, -1),
                                    )
                                    
                                    
    cn = nengo.Ensemble(n_neurons=100, dimensions=1,
                        encoders=nengo.dists.Choice([[1]]))
    
    nengo.Connection(purkinje_pause, cn)  # TODO: force inhibitory
    
    inf_olive = nengo.Ensemble(n_neurons=100, dimensions=1,
                               encoders=nengo.dists.Choice([[-1]]))
    nengo.Connection(cn, inf_olive)
    nengo.Connection(puff, inf_olive, transform=-1)
    
    c = nengo.Connection(rw.all_ensembles[0], purkinje_pause,
                         function=lambda x: 0,
                         learning_rule_type=nengo.PES(learning_rate=1e-4),
                         )
    #TODO: make this a 1:1 climbing-fibre connection
    nengo.Connection(inf_olive, c.learning_rule)
    N = 100
    #nengo.Connection(inf_olive, purkinje_pause)
    nengo.Connection(inf_olive.neurons, purkinje_pause.neurons, 
                     transform=-0.01)
    
    motor = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(cn, motor)
    
    
    
    
                         