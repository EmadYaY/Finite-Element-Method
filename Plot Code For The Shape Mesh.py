import numpy as np
import matplotlib.pyplot as plt
#The main purpose of this code is to parse and visualize a mesh.
def parse_inp_file(file_path):
    nodes = []
    elements = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        node_section = False
        element_section = False
        for line in lines:
            if '*Node' in line:
                node_section = True
                element_section = False
                continue
            if '*Element' in line:
                node_section = False
                element_section = True
                continue
            if '*Nset' in line or '*Elset' in line or line.startswith('*'):
                continue
            if node_section:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    nodes.append([float(part) for part in parts[1:]])
            if element_section:
                parts = line.strip().split(',')
                if len(parts) > 1 and all(part.strip().isdigit() for part in parts[1:]):
                    elements.append([int(part) for part in parts[1:]])
    return np.array(nodes), elements

def plot_mesh(nodes, elements):
    fig, ax = plt.subplots()
    num_nodes = nodes.shape[0]
    for element in elements:
        if all(1 <= idx <= num_nodes for idx in element):
            polygon = plt.Polygon(nodes[np.array(element) - 1], edgecolor='black', facecolor='none')
            ax.add_patch(polygon)
    ax.plot(nodes[:, 0], nodes[:, 1], 'o', markersize=2)
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mesh Plot')
    plt.show()

# مسیر فایل Mesh
file_path = 'final_2.inp'
nodes, elements = parse_inp_file(file_path)
plot_mesh(nodes, elements)
#Fazel Mohammad Ali Pour - 7/15/2024 - Python Code For Drawing the shape based on the mesh file
