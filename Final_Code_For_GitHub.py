import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def read_mesh_file(file_name):
    with open(file_name, 'r') as file:
        nodes = []
        elements = []
        read_nodes = False
        read_elements = False
        for line in file:
            if line.startswith('*Node'):
                read_nodes = True
                read_elements = False
                continue
            elif line.startswith('*Element'):
                read_nodes = False
                read_elements = True
                continue
            elif line.startswith('*'):
                read_nodes = False
                read_elements = False
                continue

            if read_nodes:
                if line.strip():
                    data = list(map(float, line.strip().split(',')))
                    nodes.append(data[1:])
            elif read_elements:
                if line.strip():
                    data = list(map(int, line.strip().split(',')))
                    elements.append(data[1:])
                    
    return np.array(nodes), np.array(elements, dtype=int)

def shape_function_derivatives(xi, eta):
    dN_dxi = 0.25 * np.array([-1 + eta, 1 - eta, 1 + eta, -1 - eta])
    dN_deta = 0.25 * np.array([-1 + xi, -1 - xi, 1 + xi, 1 - xi])
    return dN_dxi, dN_deta

def element_stiffness_and_thermal_force(x, y, conductivity_x, conductivity_y, heat_source):
    xi = [-1, 1, 1, -1]
    eta = [-1, -1, 1, 1]
    Ke = np.zeros((4, 4))
    Fe = np.zeros(4)
    for i in range(4):
        dN_dxi, dN_deta = shape_function_derivatives(xi[i], eta[i])
        J = np.array([dN_dxi, dN_deta]) @ np.column_stack((x, y))
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        B = invJ @ np.array([dN_dxi, dN_deta])
        D = np.array([[conductivity_x, 0], [0, conductivity_y]])
        Ke += B.T @ D @ B * detJ
        Fe += np.array([1, 1, 1, 1]) * heat_source * detJ / 4
    
    return Ke, Fe

def convection_boundary(x, y, convection_coefficient, ambient_temperature):
    L = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    Ke_conv = (convection_coefficient * L / 6) * np.array([[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    Fe_conv = (convection_coefficient * ambient_temperature * L / 2) * np.array([1, 1, 0, 0])
    
    return Ke_conv, Fe_conv

def apply_boundary_conditions(K, F, nodes, elements, convection_coefficient, ambient_temperature, internal_temperature):
    internal_radius = 0.01  # Internal radius (m)
    external_radius = 0.04  # External radius (m)
    internal_nodes = np.where(np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2) < (internal_radius + 0.001))[0]
    external_nodes = np.where(np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2) > (external_radius - 0.001))[0]

    # Fix the internal temperature
    for node in internal_nodes:
        K[node, :] = 0
        K[node, node] = 1
        F[node] = internal_temperature
    
    for e in range(elements.shape[0]):
        node_numbers = elements[e] - 1
        if np.any(np.isin(node_numbers, external_nodes)):
            x = nodes[node_numbers, 0]
            y = nodes[node_numbers, 1]
            Ke_conv, Fe_conv = convection_boundary(x, y, convection_coefficient, ambient_temperature)
            K[np.ix_(node_numbers, node_numbers)] += Ke_conv
            F[node_numbers] += Fe_conv
    
    return K, F

def display_results(nodes, elements, T):
    triangles = []
    for quad in elements:
        triangles.append([quad[0]-1, quad[1]-1, quad[2]-1])
        triangles.append([quad[0]-1, quad[2]-1, quad[3]-1])

    triangles = np.array(triangles)
    
    plt.figure(figsize=(8, 8))
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    plt.tricontourf(triang, T, levels=14, cmap='jet')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Temperature Distribution | <<Fazel>>')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.show()

def heat_transfer_analysis_2D(file_name):
    nodes, elements = read_mesh_file(file_name)

    conductivity_x = 2  # Thermal conductivity in x direction (W/cm°C)
    conductivity_y = 2  # Thermal conductivity in y direction (W/cm°C)
    convection_coefficient = 1.5  # Convection heat transfer coefficient (W/cm²°C)
    ambient_temperature = 20  # Ambient temperature (°C)
    internal_temperature = 140  # Internal wall temperature (°C)
    heat_source = 0  # Internal heat source (W/m³)

    num_nodes = nodes.shape[0]
    num_elements = elements.shape[0]

    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    for e in range(num_elements):
        node_numbers = elements[e] - 1
        x = nodes[node_numbers, 0]
        y = nodes[node_numbers, 1]
        
        Ke, Fe = element_stiffness_and_thermal_force(x, y, conductivity_x, conductivity_y, heat_source)
        
        K[np.ix_(node_numbers, node_numbers)] += Ke
        F[node_numbers] += Fe

    K, F = apply_boundary_conditions(K, F, nodes, elements, convection_coefficient, ambient_temperature, internal_temperature)

    T = np.linalg.solve(K, F)

    display_results(nodes, elements, T)

# Mesh file Path
file_path = r'C:\Users\argaex\Desktop\Uni\heat_transfer_final.inp'
heat_transfer_analysis_2D(file_path)
#Fazel Mohammad Ali Pour - 7/15/2024 - Python Code