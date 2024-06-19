import numpy as np
import matplotlib.pyplot as plt
#from neuron_models import Neuron, MBNeuron, KCNeuron, PNNeuron
#from connection_models import generate_sparse_connections, generate_grouped_connections


def gather_network_outputs(neurons):
    outputs = {
        "spike_trains": [neuron.has_spiked for neuron in neurons],  # Replace spike_train with has_spiked
        "membrane_potentials": [neuron.v for neuron in neurons]
    }
    return outputs

def calculate_input_current(neuron, neurons, connections):
    total_current = 0
    for conn_name, conn_matrix in connections.items():
        idx = neurons.index(neuron)
        if conn_name == 'pn_to_kc' and isinstance(neuron, KCNeuron):
            spike_amplitudes = [n.spike_amplitude for n in neurons if isinstance(n, PNNeuron) and n.has_spiked]
            print(f"Debug: spike_amplitudes (pn_to_kc) = {spike_amplitudes}, len = {len(spike_amplitudes)}")
            if len(spike_amplitudes) > 0 and idx < conn_matrix.shape[1]:
                spike_amplitudes = np.array(spike_amplitudes)[:conn_matrix.shape[0]]
                total_current += np.sum(conn_matrix[:len(spike_amplitudes), idx] * spike_amplitudes)
        elif conn_name == 'kc_to_mbon' and isinstance(neuron, MBNeuron):
            spike_amplitudes = [n.spike_amplitude for n in neurons if isinstance(n, KCNeuron) and n.has_spiked]
            print(f"Debug: spike_amplitudes (kc_to_mbon) = {spike_amplitudes}, len = {len(spike_amplitudes)}")
            if len(spike_amplitudes) > 0 and idx < conn_matrix.shape[1]:
                spike_amplitudes = np.array(spike_amplitudes)[:conn_matrix.shape[0]]
                total_current += np.sum(conn_matrix[:len(spike_amplitudes), idx] * spike_amplitudes)
        elif conn_name == 'kc_to_kc' and isinstance(neuron, KCNeuron):
            spike_amplitudes = [n.spike_amplitude for n in neurons if isinstance(n, KCNeuron) and n.has_spiked]
            print(f"Debug: spike_amplitudes (kc_to_kc) = {spike_amplitudes}, len = {len(spike_amplitudes)}")
            if len(spike_amplitudes) > 0 and idx < conn_matrix.shape[1]:
                spike_amplitudes = np.array(spike_amplitudes)[:conn_matrix.shape[0]]
                total_current += np.sum(conn_matrix[:len(spike_amplitudes), idx] * spike_amplitudes)
    return total_current

def update_neurons_and_apply_learning(neurons, connections, dt, learning=True):
    for neuron in neurons:
        input_current = calculate_input_current(neuron, neurons, connections)
        neuron.update(input_current, dt)
    if learning:
        for conn in connections.values():
            conn_shape = conn.shape
            for pre_idx in range(conn_shape[0]):
                for post_idx in range(conn_shape[1]):
                    if neurons[pre_idx].has_spiked:
                        conn[pre_idx, post_idx] *= 1.01

'''
def update_neurons_and_apply_learning(neurons, connections, dt):
    for neuron in neurons:
        input_current = calculate_input_current(neuron, neurons, connections)
        neuron.update(input_current, dt)
    for conn in connections.values():
        conn_shape = conn.shape
        for pre_idx in range(conn_shape[0]):
            for post_idx in range(conn_shape[1]):
                if neurons[pre_idx].has_spiked:
                    conn[pre_idx, post_idx] *= 1.01
'''
def run_network_simulation(input_spike_trains, simulation_duration, dt, connections, neurons):
    assert isinstance(connections, dict), "Expected connections to be a dictionary"
    assert isinstance(simulation_duration, int), "Expected simulation_duration to be an integer"
    print(f"Debug: input_spike_trains shape = {input_spike_trains.shape}")
    print(f"Debug: At simulation start, type(connections) = {type(connections)}")
    initialize_network_state(neurons)
    outputs = []
    for t in range(0, simulation_duration, dt):
        output = simulate_step(neurons, connections, input_spike_trains, t, dt)
        outputs.append(output)
    return outputs
'''
def simulate_step(neurons, connections, input_spike_trains, current_time, dt):
    print(f"Debug: In simulate_step, type(connections) = {type(connections)}")
    print(f"Debug: input_spike_trains shape = {input_spike_trains.shape}")
    apply_inputs_to_network(input_spike_trains, current_time, neurons)
    assert isinstance(connections, dict), "Connections must be a dictionary in simulate_step"
    update_neurons_and_apply_learning(neurons, connections, dt)
    return gather_network_outputs(neurons)
'''
def simulate_step(neurons, connections, input_spike_trains, current_time, dt, learning=True):
    apply_inputs_to_network(input_spike_trains, current_time, neurons)
    update_neurons_and_apply_learning(neurons, connections, dt, learning)
    return gather_network_outputs(neurons)

def apply_inputs_to_network(input_spike_trains, current_time, neurons):
    print(f"Debug: Applying inputs, current_time = {current_time}")
    for idx, neuron in enumerate(neurons):
        if idx < input_spike_trains.shape[0] and current_time < input_spike_trains.shape[1] and input_spike_trains[idx, current_time].any():
            neuron.receive_spike()
            neuron.has_spiked = True
            print(f"Debug: Neuron {idx} has spiked")


'''
def train_network(data, neurons, connections, dt, num_iterations, simulation_duration):
    for input_spike_trains in data:
        for _ in range(num_iterations):
            initialize_network_state(neurons)
            for t in range(0, simulation_duration, dt):
                output = simulate_step(neurons, connections, input_spike_trains, t, dt, learning=True)
    np.save('pn_to_kc_weights.npy', connections['pn_to_kc'])
    np.save('kc_to_kc_weights.npy', connections['kc_to_kc'])
    np.save('kc_to_mbon_weights.npy', connections['kc_to_mbon'])
    return outputs
'''
import numpy as np

def train_network(data, neurons, connections, dt, num_iterations, simulation_duration):
    for input_spike_trains in data:
        for _ in range(num_iterations):
            initialize_network_state(neurons)
            for t in range(0, simulation_duration, dt):
                output = simulate_step(neurons, connections, input_spike_trains, t, dt, learning=True)
                gather_network_outputs(neurons)
    np.save('trained_connections.npy', connections)  # Save the connections dictionary
    np.save('pn_to_kc_weights.npy', connections['pn_to_kc'])
    np.save('kc_to_kc_weights.npy', connections['kc_to_kc'])
    np.save('kc_to_mbon_weights.npy', connections['kc_to_mbon'])
    return connections



def initialize_network_state(neurons):
    for neuron in neurons:
        neuron.reset_state()
        neuron.has_spiked = False

def calculate_performance_metrics(test_results):
    if not test_results:
        return {'accuracy': 0, 'correct_predictions': 0, 'total_predictions': 0}

    correct_predictions = sum(result == expected for result, expected in test_results)
    total_predictions = len(test_results)
    accuracy = correct_predictions / total_predictions
    return {
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions
    }




def visualize_activities(neuron_activities, title):
    # Extract membrane potentials from each dictionary in neuron_activities
    membrane_potentials_list = [activity['membrane_potentials'] for activity in neuron_activities]

    # Convert the list of membrane potentials to a NumPy array and ensure it's of type float
    membrane_potentials_array = np.array(membrane_potentials_list, dtype=float)

    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Plot the membrane potentials
    plt.imshow(membrane_potentials_array, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(label='Membrane Potential')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    plt.show()