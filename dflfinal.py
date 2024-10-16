# dfl4_refined_topology_aware_modified.py

import argparse
import random
import time
from collections import defaultdict
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
# Use 'Agg' backend to prevent GUI-related errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns  # Added seaborn for enhanced visualization
from sklearn.metrics import f1_score, precision_score, recall_score
# Removed classification_report as per user request
import numpy as np
import psutil
import networkx as nx  # For network topology

# Ensure reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initialize metrics and system metrics as globals
metrics = defaultdict(lambda: defaultdict(list))
cpu_usages = []
round_times = []
node_training_times = defaultdict(list)  # To store training times per node
aggregation_times = defaultdict(list)     # To store aggregation times per round

# Command-line arguments
parser = argparse.ArgumentParser(description="Decentralized Federated Learning Simulation")
parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
parser.add_argument("--epochs", type=int, default=1, help="Number of local epochs")
parser.add_argument("--num_nodes", type=int, default=5, help="Number of nodes in the network")
parser.add_argument("--num_attackers", type=int, default=0, help="Number of attacker nodes")
parser.add_argument("--attacker_nodes", nargs='+', type=int, default=None, help="List of attacker node indices")
parser.add_argument("--attacks", nargs='+', default=[], help="Types of attacks: delay, poison")
parser.add_argument("--use_attackers", action='store_true', help="Include attacker nodes")
parser.add_argument("--participation_rate", type=float, default=0.5,
                    help="Fraction of nodes participating in each round (0 < rate <= 1)")
parser.add_argument("--topology", type=str, default="fully_connected",
                    help="Network topology: fully_connected, ring, random, custom")
parser.add_argument("--topology_file", type=str, default=None,
                    help="Path to the custom topology file (edge list). Required if topology is 'custom'.")
parser.add_argument("--max_attacks", type=int, default=None, help="Maximum number of times an attacker can perform an attack")


args = parser.parse_args()

# Post-parsing validation
if args.use_attackers and args.num_attackers == 0 and args.attacker_nodes is None:
    parser.error("--use_attackers requires --num_attackers > 0 or --attacker_nodes specified")

if args.topology == 'custom' and args.topology_file is None:
    parser.error("--topology 'custom' requires --topology_file to be specified")

# Defining the basic CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Defining the layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Define the class names for FashionMNIST
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
    
def visualize_data_distribution(train_datasets, num_nodes, class_names):
    num_classes = len(class_names)
    class_counts = np.zeros((num_nodes, num_classes))  # Assuming 10 classes in FashionMNIST

    # Aggregate class counts for each worker
    for i, dataset in enumerate(train_datasets):
        # Extract targets for the current worker
        targets = np.array(dataset.dataset.targets)[dataset.indices]
        for cls in range(num_classes):
            class_counts[i, cls] = np.sum(targets == cls)

    # Set up the plot
    plt.figure(figsize=(20, 10))
    bar_width = 0.8 / num_nodes  # Adjust bar width based on number of workers
    indices = np.arange(num_classes)  # Number of classes

    # Generate a color palette for different workers
    palette = sns.color_palette("Dark2", num_nodes)  # Use 'Set2' for better aesthetics

    # Plot bars for each worker
    for i in range(num_nodes):
        plt.bar(indices + i * bar_width, class_counts[i], width=bar_width, label=f'Worker {i}', color=palette[i])

    # Labeling and aesthetics
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title('Data Distribution Across Workers', fontsize=16)
    plt.xticks(indices + bar_width * (num_nodes / 2), class_names, rotation=45, ha='right', fontsize=12)
    plt.legend(title='Workers', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close()
    print("Data distribution plot saved as 'data_distribution.png'.")

def distribute_data_dirichlet(labels, num_nodes, alpha):
    num_classes = np.unique(labels).shape[0]
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    node_indices = {i: [] for i in range(num_nodes)}
    
    for cls in range(num_classes):
        # Shuffle the indices for the current class to ensure random distribution
        np.random.shuffle(class_indices[cls])
        
        # Sample a Dirichlet distribution for the current class
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_nodes))
        
        # Scale proportions to the number of samples in the current class
        proportions = (proportions * len(class_indices[cls])).astype(int)
        
        # Adjust proportions to ensure all samples are allocated
        # Compute the difference due to flooring
        diff = len(class_indices[cls]) - np.sum(proportions)
        for i in range(diff):
            proportions[i % num_nodes] += 1
        
        # Assign indices to each client based on the proportions
        start = 0
        for node in range(num_nodes):
            end = start + proportions[node]
            node_indices[node].extend(class_indices[cls][start:end].tolist())
            start = end
    
    return node_indices

def load_datasets_dirichlet(num_nodes, alpha):
    # Define the transformation for the dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the training and test datasets
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    # Extract labels from the training dataset
    labels = np.array(train_dataset.targets)

    # Distribute data among clients using the Dirichlet distribution
    node_indices = distribute_data_dirichlet(labels, num_nodes, alpha)

    # Create Subsets for each client based on the distributed indices
    train_datasets = [torch.utils.data.Subset(train_dataset, node_indices[i]) for i in range(num_nodes)]

    # Verification: Ensure all samples are assigned
    total_assigned = sum(len(dataset) for dataset in train_datasets)
    total_available = len(train_dataset)
    assert total_assigned == total_available, "Data assignment mismatch!"

    return train_datasets, test_dataset, labels

def print_class_distribution(train_datasets):
    for i, dataset in enumerate(train_datasets):
        labels = np.array(dataset.dataset.targets)[dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"node_{i} Class Distribution: {class_counts}")

# Create nodes
def create_nodes(num_nodes):
    nodes = list(range(num_nodes))  # Nodes are represented by integer indices
    return nodes

# Build the network topology using networkx
def build_topology(num_nodes, topology_type, topology_file=None):
    if topology_type == 'fully_connected':
        G = nx.complete_graph(num_nodes)
    elif topology_type == 'ring':
        G = nx.cycle_graph(num_nodes)
    elif topology_type == 'random':
        p_connect = 0.3  # Probability of edge creation
        G = nx.erdos_renyi_graph(num_nodes, p_connect, seed=0)
    elif topology_type == 'custom':
        if topology_file is None:
            raise ValueError("Custom topology requires a topology file.")
        G = nx.read_edgelist(topology_file, nodetype=int)
        # Ensure the graph has the correct number of nodes
        G.add_nodes_from(range(num_nodes))
    else:
        raise ValueError("Invalid topology type. Choose from 'fully_connected', 'ring', 'random', 'custom'.")
    return G

# Training function for each node
def local_train(node_id, local_model, train_dataset, epochs, attacker_type):
    # Use a fixed seed per node to ensure consistency
    node_rng = random.Random(node_id)
    device = torch.device("cpu")
    model = local_model
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    model.train()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    start_time = time.time()  # Start time for training

    for epoch in range(epochs):
        # Simulate delayed response per epoch
        if attacker_type == 'delay':
            time_delay = node_rng.uniform(30, 50)
            print(f"[node_{node_id}] Epoch {epoch+1}: Delaying computation by {time_delay:.2f} seconds.")
            time.sleep(time_delay)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Simulate model poisoning
    if attacker_type == 'poison':
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0.5)  # Add significant noise
        print(f"[node_{node_id}] Model poisoned.")

    end_time = time.time()  # End time for training
    training_time = end_time - start_time  # Calculate training time

    # Return updated model parameters and training time
    return model.state_dict(), training_time  # Return tuple

# Evaluation function
def evaluate(model, test_loader):
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    criterion = nn.NLLLoss()  # Negative log likelihood loss function
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)

    return test_loss, accuracy, f1, precision, recall

# Aggregation function using Weighted Federated Averaging (FedAvg)
def aggregate(models_state_dicts, data_sizes):
    if not models_state_dicts:
        return None

    # Initialize an empty state dict for the aggregated model
    aggregated_state_dict = {}
    total_data = sum(data_sizes)

    # Get the list of all parameter keys
    param_keys = list(models_state_dicts[0].keys())

    for key in param_keys:
        # Initialize a tensor for the weighted sum
        weighted_sum = torch.zeros_like(models_state_dicts[0][key])
        for state_dict, size in zip(models_state_dicts, data_sizes):
            weighted_sum += state_dict[key] * size
        # Compute the weighted average
        aggregated_state_dict[key] = weighted_sum / total_data

    return aggregated_state_dict

def find_nearest_participating_neighbor(G, node, participating_nodes):
    visited = set()
    queue = [node]
    visited.add(node)
    
    while queue:
        current = queue.pop(0)
        neighbors = list(G.neighbors(current))
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            if neighbor in participating_nodes:
                return neighbor
            visited.add(neighbor)
            queue.append(neighbor)
    return None

# Function to summarize model parameters
def summarize_model_parameters(node_name, model_state_dict):
    print(f"\n[Summary] Model parameters for node_{node_name} after local training:")
    for key, param in model_state_dict.items():
        param_np = param.cpu().numpy()
        mean = param_np.mean()
        std = param_np.std()
        print(f"  Layer: {key:<20} | Mean: {mean:.6f} | Std: {std:.6f}")

# Function to display and store model parameters
def display_and_store_model_parameters(node_name, model_state_dict, node_stats, rnd):
    print(f"\nModel parameters from node_{node_name} after aggregation at round {rnd}:")
    for key, param in model_state_dict.items():
        param_np = param.cpu().numpy()
        mean = param_np.mean()
        std = param_np.std()
        print(f"node_{node_name} - Layer: {key} | Mean: {mean:.6f} | Std: {std:.6f}")
        # Initialize node_stats for new nodes
        if node_name not in node_stats:
            node_stats[node_name] = {}
        if key not in node_stats[node_name]:
            node_stats[node_name][key] = {'mean': [], 'std': []}
        # Update with current round's data
        node_stats[node_name][key]['mean'].append(mean)
        node_stats[node_name][key]['std'].append(std)

# Plotting functions
def plot_metrics(metrics, rounds_range):
    # Prepare data for training metrics
    avg_metrics_train = {
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': []
    }
    # Prepare data for testing metrics
    avg_metrics_test = {
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': []
    }
    
    for rnd in rounds_range:
        acc_train, f1_train, prec_train, rec_train = [], [], [], []
        acc_test, f1_test, prec_test, rec_test = [], [], [], []
        for node in metrics:
            if len(metrics[node]['train_accuracy']) >= rnd:
                acc_train.append(metrics[node]['train_accuracy'][rnd-1])
                f1_train.append(metrics[node]['train_f1_score'][rnd-1])
                prec_train.append(metrics[node]['train_precision'][rnd-1])
                rec_train.append(metrics[node]['train_recall'][rnd-1])
            if len(metrics[node]['accuracy']) >= rnd:
                acc_test.append(metrics[node]['accuracy'][rnd-1])
                f1_test.append(metrics[node]['f1_score'][rnd-1])
                prec_test.append(metrics[node]['precision'][rnd-1])
                rec_test.append(metrics[node]['recall'][rnd-1])
        # Compute average metrics over all nodes for this round
        avg_metrics_train['Accuracy'].append(np.nanmean(acc_train) if acc_train else 0)
        avg_metrics_train['F1 Score'].append(np.nanmean(f1_train) if f1_train else 0)
        avg_metrics_train['Precision'].append(np.nanmean(prec_train) if prec_train else 0)
        avg_metrics_train['Recall'].append(np.nanmean(rec_train) if rec_train else 0)
        
        avg_metrics_test['Accuracy'].append(np.nanmean(acc_test) if acc_test else 0)
        avg_metrics_test['F1 Score'].append(np.nanmean(f1_test) if f1_test else 0)
        avg_metrics_test['Precision'].append(np.nanmean(prec_test) if prec_test else 0)
        avg_metrics_test['Recall'].append(np.nanmean(rec_test) if rec_test else 0)
    
    # Plot Training Metrics
    x = np.arange(len(rounds_range))  # the label locations
    num_metrics = 4
    width = 0.2  # Adjusted for better visibility
    
    plt.figure(figsize=(10, 6))
    # Use a colorblind-friendly palette from seaborn
    palette = sns.color_palette("Set2", num_metrics)
    
    # Compute offsets for bar positions
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    
    plt.bar(x + offsets[0], avg_metrics_train['Accuracy'], width, label='Accuracy', color=palette[0])
    plt.bar(x + offsets[1], avg_metrics_train['F1 Score'], width, label='F1 Score', color=palette[1])
    plt.bar(x + offsets[2], avg_metrics_train['Precision'], width, label='Precision', color=palette[2])
    plt.bar(x + offsets[3], avg_metrics_train['Recall'], width, label='Recall', color=palette[3])
    
    plt.ylabel('Metric Value')
    plt.xlabel('Round')
    plt.title('Average Training Metrics per Round')
    plt.xticks(x, [f'Round {rnd}' for rnd in rounds_range])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('average_training_metrics_bar_chart.png')
    plt.close()
    print("Average training metrics bar chart saved as 'average_training_metrics_bar_chart.png'.")
    
    # Plot Testing Metrics
    plt.figure(figsize=(10, 6))
    # Use a colorblind-friendly palette from seaborn
    palette = sns.color_palette("Set2", num_metrics)
    
    plt.bar(x + offsets[0], avg_metrics_test['Accuracy'], width, label='Accuracy', color=palette[0])
    plt.bar(x + offsets[1], avg_metrics_test['F1 Score'], width, label='F1 Score', color=palette[1])
    plt.bar(x + offsets[2], avg_metrics_test['Precision'], width, label='Precision', color=palette[2])
    plt.bar(x + offsets[3], avg_metrics_test['Recall'], width, label='Recall', color=palette[3])
    
    plt.ylabel('Metric Value')
    plt.xlabel('Round')
    plt.title('Average Testing Metrics per Round')
    plt.xticks(x, [f'Round {rnd}' for rnd in rounds_range])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('average_testing_metrics_bar_chart.png')
    plt.close()
    print("Average testing metrics bar chart saved as 'average_testing_metrics_bar_chart.png'.")

    # Plot Training Metrics
    plt.figure(figsize=(10, 6))
    rounds = list(rounds_range)
    # Use a colorblind-friendly palette from seaborn
    palette = sns.color_palette("Set2", 4)

    plt.plot(rounds, avg_metrics_train['Accuracy'], marker='o', label='Accuracy', color=palette[0])
    plt.plot(rounds, avg_metrics_train['F1 Score'], marker='s', label='F1 Score', color=palette[1])
    plt.plot(rounds, avg_metrics_train['Precision'], marker='^', label='Precision', color=palette[2])
    plt.plot(rounds, avg_metrics_train['Recall'], marker='d', label='Recall', color=palette[3])

    plt.ylabel('Metric Value')
    plt.xlabel('Round')
    plt.title('Average Training Metrics per Round (Line Plot)')
    plt.xticks(rounds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('average_training_metrics_line_plot.png')
    plt.close()
    print("Average training metrics line plot saved as 'average_training_metrics_line_plot.png'.")

    # Plot Testing Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, avg_metrics_test['Accuracy'], marker='o', label='Accuracy', color=palette[0])
    plt.plot(rounds, avg_metrics_test['F1 Score'], marker='s', label='F1 Score', color=palette[1])
    plt.plot(rounds, avg_metrics_test['Precision'], marker='^', label='Precision', color=palette[2])
    plt.plot(rounds, avg_metrics_test['Recall'], marker='d', label='Recall', color=palette[3])

    plt.ylabel('Metric Value')
    plt.xlabel('Round')
    plt.title('Average Testing Metrics per Round (Line Plot)')
    plt.xticks(rounds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('average_testing_metrics_line_plot.png')
    plt.close()
    print("Average testing metrics line plot saved as 'average_testing_metrics_line_plot.png'.")


def plot_loss_line(metrics, rounds_range):
    avg_loss_per_round = []
    avg_train_loss_per_round = []
    for rnd in rounds_range:
        losses = []
        train_losses = []
        for node in metrics:
            if len(metrics[node]['loss']) >= rnd:
                losses.append(metrics[node]['loss'][rnd-1])
            if len(metrics[node]['train_loss']) >= rnd:
                train_losses.append(metrics[node]['train_loss'][rnd-1])
        avg_loss = np.nanmean(losses) if losses else np.nan
        avg_train_loss = np.nanmean(train_losses) if train_losses else np.nan
        avg_loss_per_round.append(avg_loss)
        avg_train_loss_per_round.append(avg_train_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_range, avg_train_loss_per_round, marker='o', color='blue', label='Average Training Loss')
    plt.plot(rounds_range, avg_loss_per_round, marker='o', color='red', label='Average Test Loss')
    plt.title('Average Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('average_loss_over_rounds.png')
    plt.close()
    print("Average loss plot saved as 'average_loss_over_rounds.png'.")


def plot_training_aggregation_times(rounds_range, total_training_times, avg_training_times, total_aggregation_times, avg_aggregation_times):
    plt.figure(figsize=(12, 6))
    
    plt.plot(rounds_range, total_training_times, marker='o', label='Total Training Time (s)', color='darkgreen')
    #plt.plot(rounds_range, avg_training_times, marker='o', label='Average Training Time per Node (s)', color='green')
    plt.plot(rounds_range, total_aggregation_times, marker='x', label='Total Aggregation Time (s)', color='steelblue')
    #plt.plot(rounds_range, avg_aggregation_times, marker='x', label='Average Aggregation Time per Aggregation (s)', color='red')
    
    plt.title('Training and Aggregation Times per Round')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_aggregation_times_per_round.png')
    plt.close()
    print("Training and aggregation times plot saved as 'training_aggregation_times_per_round.png'.")

def plot_additional_metrics(rounds_range, cpu_usages, round_times):
    # Plot CPU Usage over Rounds
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_range, cpu_usages, marker='o', label='CPU Usage (%)', color='darkorange')
    plt.title('CPU Usage over Rounds')
    plt.xlabel('Round')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cpu_usage_over_rounds.png')
    plt.close()
    print("CPU usage plot saved as 'cpu_usage_over_rounds.png'.")

    # Plot Round Times
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_range, round_times, marker='o', label='Round Time (s)', color='purple')
    plt.title('Time Taken per Round')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('round_times_over_rounds.png')
    plt.close()
    print("Round times plot saved as 'round_times_over_rounds.png'.")

# Main simulation function
def run_simulation():
    global metrics, cpu_usages, round_times, node_training_times, aggregation_times
    num_nodes = args.num_nodes
    train_datasets, test_dataset, labels = load_datasets_dirichlet(num_nodes, alpha=0.5)  
    print_class_distribution(train_datasets)
    
    visualize_data_distribution(train_datasets, num_nodes, FASHION_MNIST_CLASSES)
    nodes = create_nodes(num_nodes)

    # Build network topology
    G = build_topology(num_nodes, args.topology, args.topology_file)

    # Visualize and save the network topology
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=0)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    #labels_graph = {i: f"node_{i}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(f"Network Topology: {args.topology}")
    plt.savefig('network_topology.png')
    plt.close()
    print("Network topology plot saved as 'network_topology.png'.")

    # Map datasets to nodes
    node_datasets = {node: train_datasets[node] for node in nodes}

    # Split the global test_dataset into per-node test datasets
    # Shuffle and split test dataset equally among nodes
    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)
    split_size = len(test_dataset) // num_nodes
    test_loaders_per_node = {}
    for i in range(num_nodes):
        start = i * split_size
        end = (i + 1) * split_size if i < num_nodes -1 else len(test_dataset)
        node_test_indices = test_indices[start:end]
        node_test_subset = torch.utils.data.Subset(test_dataset, node_test_indices)
        node_test_loader = torch.utils.data.DataLoader(node_test_subset, batch_size=32, shuffle=False)
        test_loaders_per_node[nodes[i]] = node_test_loader

    random.seed(42)  # Fixed seed for participant selection
    np.random.seed(42)

    # Precompute participating nodes for each round
    num_rounds = args.rounds
    participating_nodes_per_round = []
    num_participants = max(1, int(num_nodes * args.participation_rate))

    # Use a fixed seed for participation selection
    participation_rng = random.Random(42)  # Use any fixed number
    for rnd in range(1, num_rounds + 1):
        participating_nodes = participation_rng.sample(nodes, num_participants)
        participating_nodes_per_round.append(participating_nodes)

    # Identify attacker nodes
    attacker_node_ids = []
    attack_counts = {}
    if args.use_attackers:
        if args.attacker_nodes is not None:
            attacker_node_ids = args.attacker_nodes
        else:
            # Ensure num_attackers does not exceed num_nodes
            num_attackers = min(args.num_attackers, num_nodes)
            # Use fixed seed random generator for attacker selection
            attacker_rng = random.Random(12345)
            attacker_node_ids = attacker_rng.sample(range(num_nodes), num_attackers)
        print(f"Attacker nodes: {attacker_node_ids} with attacks: {args.attacks}")
        # Initialize attack counts for each attacker
        for attacker_node in attacker_node_ids:
                attack_counts[attacker_node] = 0  # Initialize attack count to 0


    # Initialize local models for each node
    local_models = {}
    for node in nodes:
        local_models[node] = Net()
    local_params = {node: local_models[node].state_dict() for node in nodes}

    # Store node stats for parameter evolution
    node_stats = {}
    rounds_list = []  # Initialize rounds list

    # Use ThreadPoolExecutor for simulating concurrent execution
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), num_nodes)) as executor:
        for rnd in range(1, args.rounds + 1):
            print(f"\n=== Round {rnd}/{args.rounds} ===")
            rounds_list.append(rnd)  # Append current round number
            start_time_round = time.time()
            cpu_usage = psutil.cpu_percent(interval=None)  # Record CPU usage at the start

            # Use precomputed participating nodes
            participating_nodes = participating_nodes_per_round[rnd - 1]
            print(f"Participating nodes: {participating_nodes}")

            # Local training for each participating node
            future_to_node = {}
            for node in participating_nodes:
                attacker_type = None
                if node in attacker_node_ids:
                    # Determine if the attacker will perform an attack in this round
                    max_attacks = args.max_attacks
                    performed_attacks = attack_counts[node]
                    if max_attacks is None or performed_attacks < max_attacks:
                    # Attacker will perform an attack
                        if args.attacks:
                            attacker_type = random.choice(args.attacks)
                            attack_counts[node] += 1  # Increment attack count
                            print(f"Node {node} is performing '{attacker_type}' attack.")
                    else:
                        # Attacker will not perform an attack
                        print(f"Node {node} is an attacker but will not perform attack in this round (attack limit reached).")
                # Load the local model
                local_model = local_models[node]
                # Submit local training task
                future = executor.submit(local_train, node, local_model, node_datasets[node], args.epochs, attacker_type)
                future_to_node[future] = node

            # Collect updated local models and training times
            successful_nodes = []
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    updated_params, training_time = future.result()  # Receive tuple
                    # Update the local model parameters
                    local_models[node].load_state_dict(updated_params)
                    successful_nodes.append(node)
                    print(f"Node {node} completed training in {training_time:.2f} seconds.")  # Print training time

                    summarize_model_parameters(node, updated_params)

                    # Record training time
                    node_training_times[node].append(training_time)  # Record training time

                except Exception as exc:
                    print(f"Node {node} generated an exception during training: {exc}")

            # Simulate communication and aggregation among participating nodes
            # Each participating node only communicates with reachable participating nodes
            if successful_nodes:
                # Build subgraph of G with participating nodes
                G_participating = G.subgraph(successful_nodes)

                # Find connected components in G_participating
                components = list(nx.connected_components(G_participating))

                for component in components:
                    component_nodes = list(component)
                    print(f"Aggregating models for connected component: {component_nodes}")

                    # Collect models and data sizes
                    component_models = [local_models[node].state_dict() for node in component_nodes]
                    component_data_sizes = [len(node_datasets[node]) for node in component_nodes]

                    # Measure aggregation time
                    start_time_agg = time.time()  # Start time for aggregation

                    # Perform aggregation
                    aggregated_params = aggregate(component_models, component_data_sizes)

                    end_time_agg = time.time()  # End time for aggregation
                    aggregation_time = end_time_agg - start_time_agg  # Calculate aggregation time

                    # Record aggregation time
                    aggregation_times[rnd].append(aggregation_time)  # Record aggregation time

                    if aggregated_params is not None:
                        # Update all nodes in the component with the aggregated model
                        for node in component_nodes:
                            local_models[node].load_state_dict(aggregated_params)
                            display_and_store_model_parameters(node, aggregated_params, node_stats, rnd)
                    else:
                        print(f"No aggregation occurred for component: {component_nodes}")

                    print(f"Aggregation for component {component_nodes} took {aggregation_time:.4f} seconds.")  # Print aggregation time with higher precision

            else:
                print("No participating nodes completed training this round.")

            # Non-participating nodes receive the latest model from their nearest participating neighbors
            non_participating_nodes = set(nodes) - set(participating_nodes)

            # Aggregate and update non-participating nodes based on their nearest participating neighbors
            for node in non_participating_nodes:
                # Find the nearest participating neighbor
                nearest_participating = find_nearest_participating_neighbor(G, node, participating_nodes)
                if nearest_participating is not None:
                    # Fetch the model from the nearest participating neighbor
                    neighbor_model = local_models[nearest_participating].state_dict()
                    # Update the non-participating node's model
                    local_models[node].load_state_dict(neighbor_model)
                    display_and_store_model_parameters(node, neighbor_model, node_stats, rnd)
                    print(f"Node {node} receives update from nearest participating neighbor: {nearest_participating}")
                else:
                    print(f"Node {node} has no participating neighbors to receive updates from and retains its current model.")


                # Evaluate each node's model on its own training data
            print("\n=== Evaluation Phase of training data ===")
            for node in nodes:
                train_loader = torch.utils.data.DataLoader(node_datasets[node], batch_size=32, shuffle=False)
                model = local_models[node]
                print(f"\nEvaluating model for node_{node} on training data...")
                train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader)
                print(f"[Round {rnd}] Training Data Evaluation of node_{node} -> Loss: {train_loss:.4f}, "
                        f"Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, "
                        f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                # Record training data metrics per node
                metrics[node]['train_loss'].append(train_loss)
                metrics[node]['train_accuracy'].append(train_accuracy)
                metrics[node]['train_f1_score'].append(train_f1)
                metrics[node]['train_precision'].append(train_precision)
                metrics[node]['train_recall'].append(train_recall)


            # Evaluate each node's model on its own test data
            print("\n=== Evaluation Phase of testing data ===")
            for node in nodes:
                test_loader = test_loaders_per_node[node]
                model = local_models[node]
                print(f"\nEvaluating model for node_{node} on testing data...")
                loss, accuracy, f1, precision, recall = evaluate(model, test_loader)
                print(f"[Round {rnd}] Evaluation of node_{node} -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                      f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                # Record metrics per node
                metrics[node]['loss'].append(loss)
                metrics[node]['accuracy'].append(accuracy)
                metrics[node]['f1_score'].append(f1)
                metrics[node]['precision'].append(precision)
                metrics[node]['recall'].append(recall)
            # **Modified Evaluation Phase End ***

            end_time_round = time.time()
            round_time = end_time_round - start_time_round
            round_times.append(round_time)
            cpu_usages.append(cpu_usage)
            print(f"[Round {rnd}] Time taken: {round_time:.2f} seconds")

def main_with_plot():
    try:
        run_simulation()

        # After all rounds, compute training and aggregation times per round
        rounds_range = range(1, args.rounds + 1)

        # Initialize lists to store total and average times per round
        total_training_time_per_round = []
        avg_training_time_per_round = []
        total_aggregation_time_per_round = []
        avg_aggregation_time_per_round = []

        for rnd in rounds_range:
            # Calculate total training time for the round
            total_training_time = sum(node_training_times[node][rnd-1] for node in node_training_times if len(node_training_times[node]) >= rnd)
            avg_training_time = total_training_time / args.num_nodes if args.num_nodes > 0 else 0
            total_training_time_per_round.append(total_training_time)
            avg_training_time_per_round.append(avg_training_time)

            # Calculate total aggregation time for the round
            total_aggregation_time = sum(aggregation_times[rnd]) if rnd in aggregation_times else 0
            # Calculate average aggregation time per aggregation in the round
            num_aggregations = len(aggregation_times[rnd]) if rnd in aggregation_times else 1  # Avoid division by zero
            avg_aggregation_time = total_aggregation_time / num_aggregations if num_aggregations > 0 else 0
            total_aggregation_time_per_round.append(total_aggregation_time)
            avg_aggregation_time_per_round.append(avg_aggregation_time)

        # Plot the metrics
        plot_metrics(metrics, rounds_range)  # Existing plot
        plot_loss_line(metrics, rounds_range)           # Existing plot
        plot_training_aggregation_times(rounds_range, total_training_time_per_round, avg_training_time_per_round,
                                       total_aggregation_time_per_round, avg_aggregation_time_per_round)  # New plot
        plot_additional_metrics(rounds_range, cpu_usages, round_times)  # Existing plot

        # Calculate and print detailed statistics
        if cpu_usages and round_times:
            avg_cpu_usage = np.mean(cpu_usages)
            #std_cpu_usage = np.std(cpu_usages)
            avg_round_time = np.mean(round_times)
            #std_round_time = np.std(round_times)
            print(f"\nAverage CPU Usage per Round: {avg_cpu_usage:.2f}%")
            print(f"Average Time Taken per Round: {avg_round_time:.2f} seconds")

            # Calculate total metrics
            total_cpu_usage = np.sum(cpu_usages)
            total_round_time = np.sum(round_times)
            #print(f"Total CPU Usage across all Rounds: {total_cpu_usage:.2f}%")
            #print(f"Total Time Taken across all Rounds: {total_round_time:.2f} seconds")

            # Calculate total and average training times
            #total_training_time = sum(total_training_time_per_round)
            avg_training_time = np.mean(total_training_time_per_round) if args.rounds > 0 else 0
            #print(f"Total Training Time across all Rounds: {total_training_time:.2f} seconds")
            print(f"Average Training Time per Round: {avg_training_time:.2f} seconds")

            # Calculate total and average aggregation times
            #total_aggregation_time = sum(total_aggregation_time_per_round)
            avg_aggregation_time = np.mean(total_aggregation_time_per_round) if args.rounds > 0 else 0
            #print(f"Total Aggregation Time across all Rounds: {total_aggregation_time:.4f} seconds")
            print(f"Average Aggregation Time per Round: {avg_aggregation_time:.4f} seconds")
        else:
            print("\nNo metrics recorded to compute averages.")
        
        # Compute average evaluation metrics over all nodes and rounds
        total_test_losses = []
        total_accuracies = []
        total_f1_scores = []
        total_precisions = []
        total_recalls = []

        for node in metrics:
            total_test_losses.extend(metrics[node]['loss'])
            total_accuracies.extend(metrics[node]['accuracy'])
            total_f1_scores.extend(metrics[node]['f1_score'])
            total_precisions.extend(metrics[node]['precision'])
            total_recalls.extend(metrics[node]['recall'])

        # Now compute the averages
        avg_test_loss = np.mean(total_test_losses)
        avg_accuracy = np.mean(total_accuracies)
        avg_f1_score = np.mean(total_f1_scores)
        avg_precision = np.mean(total_precisions)
        avg_recall = np.mean(total_recalls)

        # Print the averages
        print("\nAverage Evaluation Metrics over all nodes and rounds:")
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average F1 Score: {avg_f1_score:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")


        print("\nSimulation complete. Plots have been saved as PNG files.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main_with_plot()
