import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def read_mesh_file(file_name):
    with open(file_name, 'r') as file:
        nodes = []
        elements = []
        node_sets = {}
        element_sets = {}
        
        current_set = ''
        lines = file.readlines()
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if '*Node' in line:
                i += 1
                line = lines[i].strip()
                while not line.startswith('*'):
                    node_data = [float(val) for val in line.split(',') if val]
                    nodes.append(node_data)
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
            elif '*Element' in line:
                i += 1
                line = lines[i].strip()
                while not line.startswith('*'):
                    element_data = [int(val) for val in line.split(',') if val]
                    elements.append(element_data)
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
            elif '*Nset' in line:
                current_set = line.split('=')[1].strip()
                current_set = current_set.replace('-', '_')
                node_sets[current_set] = []
                i += 1
                line = lines[i].strip()
                while not line.startswith('*'):
                    node_set_data = [int(val) for val in line.split(',') if val]
                    node_sets[current_set].extend(node_set_data)
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
            elif '*Elset' in line:
                current_set = line.split('=')[1].strip()
                current_set = current_set.replace('-', '_')
                element_sets[current_set] = []
                i += 1
                line = lines[i].strip()
                while not line.startswith('*'):
                    element_set_data = [int(val) for val in line.split(',') if val]
                    element_sets[current_set].extend(element_set_data)
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
            else:
                i += 1

    return np.array(nodes), np.array(elements), node_sets, element_sets

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

def apply_boundary_conditions(K, F, node_sets, convection_coefficient, ambient_temperature, internal_temperature):
    internal_nodes = np.array(node_sets['Set_in']) - 1
    external_nodes = np.array(node_sets['Set_out']) - 1

    # Fix the internal temperature
    for node in internal_nodes:
        K[node, :] = 0
        K[node, node] = 1
        F[node] = internal_temperature
    
    for node in external_nodes:
        K[node, node] += convection_coefficient
        F[node] += convection_coefficient * ambient_temperature
    
    return K, F

def display_results(nodes, elements, T):
    triangles = []
    for quad in elements:
        triangles.append([quad[1]-1, quad[2]-1, quad[3]-1])
        triangles.append([quad[1]-1, quad[3]-1, quad[4]-1])

    triangles = np.array(triangles)
    
    plt.figure(figsize=(8, 8))
    triang = tri.Triangulation(nodes[:, 1], nodes[:, 2], triangles)
    contour = plt.tricontourf(triang, T, levels=14, cmap='jet')
    plt.colorbar(contour, label='Temperature (°C)')
    contour_lines = plt.tricontour(triang, T, levels=14, colors='black', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8)
    plt.title('Temperature Distribution | <<Fazel>>')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.show()

def heat_transfer_analysis_2D(file_name):
    nodes, elements, node_sets, element_sets = read_mesh_file(file_name)

    conductivity_x = 2  # Thermal conductivity in x direction (W/cm°C)
    conductivity_y = 2  # Thermal conductivity in y direction (W/cm°C)
    convection_coefficient = 1.5  # Convection heat transfer coefficient (W/cm²°C)
    ambient_temperature = 20  # Ambient temperature (°C)
    internal_temperature = 140  # Internal wall temperature (°C)
    heat_source = 0  # Internal heat source (W/m³)

    num_nodes = nodes.shape[0]
    num_elements = elements.shape[0]

    K = sp.lil_matrix((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    for e in range(num_elements):
        node_numbers = elements[e, 1:] - 1
        x = nodes[node_numbers, 1]
        y = nodes[node_numbers, 2]
        
        Ke, Fe = element_stiffness_and_thermal_force(x, y, conductivity_x, conductivity_y, heat_source)
        
        for i in range(4):
            for j in range(4):
                K[node_numbers[i], node_numbers[j]] += Ke[i, j]
            F[node_numbers[i]] += Fe[i]

    K, F = apply_boundary_conditions(K, F, node_sets, convection_coefficient, ambient_temperature, internal_temperature)

    T = spla.spsolve(K.tocsr(), F)

    display_results(nodes, elements, T)

# Mesh file Path
file_path = 'C:/Users/argaex/Desktop/Uni/heat_transfer_final.inp'
heat_transfer_analysis_2D(file_path)


#Fazel Mohammad Ali Pour - 7/15/2024 - Python Code - Updated