import nengo
import numpy as np
import scipy.stats

class EyeblinkReflex(nengo.Process):
    """
    This class implements the trajectory generation for the agonist and
    antagonist muscles generating an eyeblink.
    """

    P_AGONIST    = [ 0.29104462, -0.5677302,  0.0125624,  81.2206584]
    T_AGONIST    = 250e-3
    P_ANTAGONIST = [ 0.33981309, 10.7624779,  0.3168123,   3.4498503]
    T_ANTAGONIST = 275e-3

    def __init__(self):
        super().__init__(default_size_in=1, default_size_out=2)

    @staticmethod
    def skewnorm(x, mu, a, std, s):
        return (s * std) * scipy.stats.skewnorm.pdf(x, a, mu, std)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        # Compute the convolutions
        def mkconv(P, T0, T1=1.0):
            ts = np.arange(T0, T1, dt)
            return EyeblinkReflex.skewnorm(ts, *P)

        conv_ag = mkconv(
            EyeblinkReflex.P_AGONIST, EyeblinkReflex.T_AGONIST)
        conv_an = mkconv(
            EyeblinkReflex.P_ANTAGONIST, EyeblinkReflex.T_ANTAGONIST)

        state=np.zeros(1 + len(conv_ag) + len(conv_an))
        I_lx = slice(0, 1)                                 # last x
        I_ag = slice(I_lx.stop, I_lx.stop + len(conv_ag))  # agonist
        I_an = slice(I_ag.stop, I_ag.stop + len(conv_an))  # antagonist

        def step(t, x, state=state):
            # Compute the differential, compute a positive and negative branch
            # of the differential
            lx, ag, an = state[I_lx], state[I_ag], state[I_an]
            dx = (x - lx[0]) / dt
            dxp, dxn = np.maximum(0, dx), np.maximum(0, -dx)

            # Store the last x value
            lx[0] = x

            # Shift the history by one element and insert the new
            # agonist/antagonist
            ag[1:] = ag[:-1]
            an[1:] = an[:-1]
            ag[0] = dxp
            an[0] = dxn
            
            # Return the current state as output
            return np.array((
                np.sum(conv_ag * ag) * dt,
                np.sum(conv_an * an) * dt))

        return step



class Eyelid(nengo.Process):
    """
    The Eyelid class models the eylid. It turns an agonist and antagonist input
    into an eyelid "closedness" between zero (full closed) and one (full open).
    """

    def __init__(self):
        super().__init__(default_size_in=2, default_size_out=1)
        self.step = self.make_step(2, 1, 1e-3, None, None)


    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state=np.zeros(1)
        state[0] = 0.0
        

        def step(t, x, state=state):
            # Update the eyelid location
            state[0] = np.clip(state[0] + (np.clip(x[0], 0, None) -
                                          (np.clip(x[1], 0, None))) * dt, 0, 1)

            # Return the state as output
            return np.array((state[0],))

        setattr(step, 'state', state);

        return step

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    @property
    def _nengo_html_(self):
        return """
<svg width="200" height="100" viewBox="-1.9 -1.5 3.5 2.75" version="1.1">
	<defs>
		<clipPath clipPathUnits="userSpaceOnUse" id="clipPath973">
			<path style="fill:#ffffff" d="M -2.3,0 C -2.3,0 -0.9,{o} 0,{o} 1,{o} 2.3,0 2.3,0 2.3,0 1,-{o} 0,-{o} -1,-{o} -2.3,0 -2.3,0 Z" />
		</clipPath>
	</defs>
	<path
		style="fill:#ffffff;stroke:#000000;stroke-width:0.04px"
		d="M -2.3,0 C -2.3,0 -0.9,{o} 0,{o} 1,{o} 2.3,0 2.3,0 2.3,0 1,-{o} 0,-{o} -1,-{o} -2.3,0 -2.3,0 Z" />
	<path
		style="fill:#000000"  clip-path="url(#clipPath973)"
		d="m 0.00568441,-0.83354084 a 0.80501685,0.80501685 0 0 0 -0.80511881,0.8051188 0.80501685,0.80501685 0 0 0 0.80511881,0.80511881 0.80501685,0.80501685 0 0 0 0.8051188,-0.80511881 0.80501685,0.80501685 0 0 0 -0.8051188,-0.8051188 z m 0,0.57877603 a 0.22641097,0.22641097 0 0 1 0.22634277,0.22634277 0.22641097,0.22641097 0 0 1 -0.22634277,0.22634277 0.22641097,0.22641097 0 0 1 -0.22634277,-0.22634277 0.22641097,0.22641097 0 0 1 0.22634277,-0.22634277 z" />
</svg>""".format(o=1.0 - self.step.state[0])

