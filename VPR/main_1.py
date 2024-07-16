from connection_models import generate_sparse_connections, generate_grouped_connections


folder_path = "/content/drive/MyDrive/Colab Notebooks/Robotics_capstone/VPR_Using_SNN_(Scirobotics)/OneDrive_1_28-08-2023/dvs_frames/p01"
images = load_images_from_folder(folder_path)
processed_images = preprocess_images(images)
differences = calculate_frame_differences(processed_images)
normalized_diffs = normalize_differences(differences)
input_spike_trains = convert_to_spike_trains(normalized_diffs)

num_pns = 434
num_kcs = 300
num_mbons = 50

#pn_neurons = [PNNeuron() for _ in range(num_pns)]
#kc_neurons = [KCNeuron() for _ in range(num_kcs)]
#mbon_neurons = [MBNeuron() for _ in range(num_mbons)]

neurons = [PNNeuron() for _ in range(num_pns)] + [KCNeuron() for _ in range(num_kcs)] + [MBNeuron() for _ in range(num_mbons)]

pn_to_kc_connections = generate_sparse_connections(num_pns, num_kcs)
kc_to_mbon_connections = generate_sparse_connections(num_kcs, num_mbons)
kc_to_kc_connections = generate_grouped_connections(num_kcs, num_kcs, num_groups=5)

connections = {
    'pn_to_kc': pn_to_kc_connections,
    'kc_to_kc': kc_to_kc_connections,
    'kc_to_mbon': kc_to_mbon_connections
}

dt = 1
simulation_duration = input_spike_trains.shape[1]

print(f"Debug: Before calling train_network, type(connections) = ", type(connections))
print("Debug: Before calling train_network, connections content = ", connections)

outputs = train_network([input_spike_trains], neurons, connections, dt, num_iterations=1, simulation_duration=simulation_duration)

performance_metrics = calculate_performance_metrics(outputs)
print("Performance metrics:", performance_metrics)
