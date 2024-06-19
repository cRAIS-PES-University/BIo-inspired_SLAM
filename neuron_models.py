class Neuron:
    def __init__(self, **kwargs):
        # Standard properties
        self.v_rest = kwargs.get('v_rest', -70)
        self.cm = kwargs.get('cm', 1.0)
        self.tau_m = kwargs.get('tau_m', 20.0)
        self.tau_refrac = kwargs.get('tau_refrac', 2.0)
        self.tau_syn_E = kwargs.get('tau_syn_E', 5.0)
        self.tau_syn_I = kwargs.get('tau_syn_I', 5.0)
        self.v_thresh = kwargs.get('v_thresh', -40.0)
        self.v_reset = kwargs.get('v_reset', -80.0)
        self.i_offset = kwargs.get('i_offset', 0.0)
        self.spike_amplitude = kwargs.get('spike_amplitude', 1.0)  # Added spike_amplitude
        # Additional dynamics for adaptation
        self.tau_ca2 = kwargs.get('tau_ca2', 100.0)
        self.i_ca2 = kwargs.get('i_ca2', 0.0)
        self.i_alpha = kwargs.get('i_alpha', 0.0)
        self.v = self.v_rest
        self.ca2_concentration = 0.0
        self.has_spiked = False

    def update(self, input_current, dt):
        self.has_spiked = False
        dv = (-(self.v - self.v_rest) + input_current * self.cm + self.i_offset - self.ca2_concentration * self.i_alpha) / self.tau_m
        self.v += dv * dt
        self.ca2_concentration += (-self.ca2_concentration / self.tau_ca2 + self.i_ca2) * dt
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.has_spiked = True
        return self.has_spiked

    def reset_state(self):
        self.v = self.v_rest
        self.ca2_concentration = 0

    def receive_spike(self):
        self.v += self.tau_syn_E

class MBNeuron(Neuron):
    def __init__(self):
        super().__init__(v_rest=-70.0, cm=1.0, tau_m=20.0, tau_refrac=2.0, tau_syn_E=5.0, tau_syn_I=5.0, v_thresh=-50.0, v_reset=-65.0, i_offset=0.01, tau_ca2=20.0, i_ca2=0.0, i_alpha=0.1)

class KCNeuron(Neuron):
    def __init__(self):
        super().__init__(v_rest=-65.0, cm=0.9, tau_m=15.0, tau_refrac=1.5, tau_syn_E=3.0, tau_syn_I=2.0, v_thresh=-52.0, v_reset=-68.0, i_offset=0.0)

class PNNeuron(Neuron):
    def __init__(self):
        super().__init__(v_rest=-68.0, cm=1.1, tau_m=25.0, tau_refrac=2.5, tau_syn_E=6.0, tau_syn_I=6.0, v_thresh=-54.0, v_reset=-70.0, i_offset=0.01, tau_ca2=20.0, i_ca2=0.0, i_alpha=0.1)
