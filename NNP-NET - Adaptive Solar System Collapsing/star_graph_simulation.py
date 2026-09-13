def simulate_fast_subnetwork_iteration(num_nodes=50):
    # Create a star graph
    # Node 0 is the hub, Nodes 1 to num_nodes-1 are outskirt nodes
    edges = {i: {} for i in range(num_nodes)}
    for i in range(1, num_nodes):
        edges[0][i] = 1.0
        edges[i][0] = 1.0

    print("Initial graph: Star graph with 1 hub and", num_nodes - 1, "outskirt nodes. All edge weights = 1.0\n")

    current_node_count = num_nodes
    target = 10
    
    iteration = 1
    
    while current_node_count > target:
        # Array equivalent to partOfNode
        part_of_node = {i: -1 for i in range(num_nodes)}
        dist_from_center = {i: 0.0 for i in range(num_nodes)}
        out_edges = {}
        out_node_count = 0
        
        # Nodes are processed lowest degree first.
        # Outskirt nodes have degree 1, hub node has degree N-1.
        degrees = [(len(edges[i]), i) for i in range(num_nodes)]
        degrees.sort() # Sorts lowest degree first
        
        for deg, i in degrees:
            if part_of_node[i] == -1:
                # Merge unassigned neighbors
                part_of_node[i] = out_node_count
                count = 0
                for neighbor, weight in edges[i].items():
                    if part_of_node[neighbor] == -1 and neighbor != i:
                        part_of_node[neighbor] = out_node_count
                        dist_from_center[neighbor] = weight
                        count += 1
                
                current_node_count -= count
                out_edges[out_node_count] = {}
                out_node_count += 1
                
                if current_node_count <= target:
                    break
        
        # Add everything that has not been assigned yet
        for i in range(num_nodes):
            if part_of_node[i] == -1:
                part_of_node[i] = out_node_count
                out_edges[out_node_count] = {}
                out_node_count += 1
        
        # Calculate new edges based on C++ logic
        for i in range(num_nodes):
            for neighbor, weight in edges[i].items():
                if part_of_node[i] == part_of_node[neighbor]:
                    continue
                
                dist = dist_from_center[i] + dist_from_center[neighbor] + weight
                
                u = part_of_node[i]
                v = part_of_node[neighbor]
                
                if v not in out_edges[u] or out_edges[u][v] > dist:
                    out_edges[u][v] = dist
        
        # Overwrite edges for next iteration
        edges = out_edges
        num_nodes = out_node_count
        
        # Find max edge weight
        max_weight = 0
        for u in edges:
            for v, w in edges[u].items():
                if w > max_weight:
                    max_weight = w
                    
        print(f"Iteration {iteration}: Node count = {num_nodes}, Max edge weight = {max_weight}")
        iteration += 1

if __name__ == '__main__':
    simulate_fast_subnetwork_iteration(30)
