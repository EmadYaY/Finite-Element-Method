% Main script for heat transfer analysis
% Define file name and read mesh data
fileName = 'heat_transfer_final.inp';
[nodes, elements, node_sets, element_sets] = readMeshFile(fileName);

% Defining problem parameters
conductivity_x = 2; % Thermal conductivity in x direction (W/cm°C)
conductivity_y = 2; % Thermal conductivity in y direction (W/cm°C)
convectionCoefficient = 1.5; % Convective heat transfer coefficient (W/cm²°C)
ambientTemperature = 20; % Ambient temperature (°C)
internalTemperature = 140; % Internal wall temperature (°C)
heatSource = 0; % Internal heat source (W/m³)

% Number of nodes and elements
numNodes = size(nodes, 1);
numElements = size(elements, 1);

% Creating global stiffness matrix and thermal force vector
K = sparse(numNodes, numNodes);
F = zeros(numNodes, 1);

% Calculating stiffness matrix and thermal force vector for each element
for e = 1:numElements
    nodeNumbers = elements(e, 2:end);
    x = nodes(nodeNumbers, 2);
    y = nodes(nodeNumbers, 3);
    
    [Ke, Fe] = elementStiffnessAndThermalForce(x, y, conductivity_x, conductivity_y, heatSource);
    
    K(nodeNumbers, nodeNumbers) = K(nodeNumbers, nodeNumbers) + Ke;
    F(nodeNumbers) = F(nodeNumbers) + Fe;
end

% Applying boundary conditions
[K, F] = applyBoundaryConditions(K, F, node_sets, convectionCoefficient, ambientTemperature, internalTemperature);

% Solving the system of equations
T = K \ F;

% Displaying results
displayResults(nodes, elements, T);

function [Ke, Fe] = elementStiffnessAndThermalForce(x, y, conductivity_x, conductivity_y, heatSource)
    % Calculating the B matrix for a four-node element
    xi = [-1 1 1 -1];
    eta = [-1 -1 1 1];
    Ke = zeros(4,4);
    Fe = zeros(4,1);
    
    for i = 1:4
        [dN_dxi, dN_deta] = shapeFunction(xi(i), eta(i));
        J = [dN_dxi; dN_deta] * [x, y];
        detJ = det(J);
        invJ = inv(J);
        B = invJ * [dN_dxi; dN_deta];
        
        D = [conductivity_x 0; 0 conductivity_y];
        
        Ke = Ke + B' * D * B * detJ;
        Fe = Fe + [1; 1; 1; 1] * heatSource * detJ / 4;
    end
end

function [dN_dxi, dN_deta] = shapeFunction(xi, eta)
    dN_dxi = 0.25 * [-1+eta, 1-eta, 1+eta, -1-eta];
    dN_deta = 0.25 * [-1+xi, -1-xi, 1+xi, 1-xi];
end

function [K, F] = applyBoundaryConditions(K, F, node_sets, convectionCoefficient, ambientTemperature, internalTemperature)
    % Identifying boundary nodes
    internal_nodes = node_sets.Set_in;
    external_nodes = node_sets.Set_out;
    
    % Applying fixed temperature boundary condition at internal boundary
    K(internal_nodes, :) = 0;
    K(internal_nodes, internal_nodes) = eye(length(internal_nodes));
    F(internal_nodes) = internalTemperature;
    
    % Applying convective heat transfer boundary condition at external boundary
    for i = 1:length(external_nodes)
        node = external_nodes(i);
        K(node, node) = K(node, node) + convectionCoefficient;
        F(node) = F(node) + convectionCoefficient * ambientTemperature;
    end
end

function displayResults(nodes, elements, T)
    figure;
    patch('Faces', elements(:,2:end), 'Vertices', nodes(:,2:3), 'FaceVertexCData', T, 'FaceColor', 'interp', 'EdgeColor', 'none');
    colorbar;
    title('Temperature Distribution - <<Fazel>>');
    xlabel('X [m]');
    ylabel('Y [m]');
    axis equal;
    colormap(jet);
end

function [nodes, elements, node_sets, element_sets] = readMeshFile(fileName)
    fid = fopen(fileName, 'r');
    nodes = [];
    elements = [];
    node_sets = struct();
    element_sets = struct();
    
    current_nset = '';
    current_elset = '';
    line = fgetl(fid);
    
    while ischar(line)
        if contains(line, '*Node')
            line = fgetl(fid);
            while ~startsWith(line, '*')
                nodeData = sscanf(line, '%f,%f,%f,%f');
                nodes = [nodes; nodeData'];
                line = fgetl(fid);
                if ~ischar(line)
                    break;
                end
            end
        elseif contains(line, '*Element')
            line = fgetl(fid);
            while ~startsWith(line, '*')
                elementData = sscanf(line, '%f,%f,%f,%f,%f');
                elements = [elements; elementData'];
                line = fgetl(fid);
                if ~ischar(line)
                    break;
                end
            end
        elseif contains(line, '*Nset')
            current_nset = extractAfter(line, 'nset=');
            current_nset = strtrim(current_nset);
            current_nset = matlab.lang.makeValidName(current_nset); % Sanitize the field name
            node_sets.(current_nset) = [];
            line = fgetl(fid);
            while ~startsWith(line, '*')
                nodeSetsData = sscanf(line, '%d,');
                node_sets.(current_nset) = [node_sets.(current_nset), nodeSetsData'];
                line = fgetl(fid);
                if ~ischar(line)
                    break;
                end
            end
        elseif contains(line, '*Elset')
            current_elset = extractAfter(line, 'elset=');
            current_elset = strtrim(current_elset);
            current_elset = matlab.lang.makeValidName(current_elset); % Sanitize the field name
            element_sets.(current_elset) = [];
            line = fgetl(fid);
            while ~startsWith(line, '*')
                elementSetsData = sscanf(line, '%d,');
                element_sets.(current_elset) = [element_sets.(current_elset), elementSetsData'];
                line = fgetl(fid);
                if ~ischar(line)
                    break;
                end
            end
        else
            line = fgetl(fid);
        end
    end
    
    fclose(fid);
end
%Fazel Mohammad Ali Pour - 7/15/2024 - Matlab Code