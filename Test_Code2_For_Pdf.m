function heat_transfer_analysis_2D()
    % Main function to perform 2D heat transfer analysis using FEM

    % Reading mesh file data
    fileName = 'C:\Users\argaex\Desktop\Uni\final_2.inp';
    [nodes, elements] = readMeshFile(fileName);

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
    [K, F] = applyBoundaryConditions(K, F, nodes, elements, convectionCoefficient, ambientTemperature, internalTemperature);

    % Solving the system of equations
    T = K \ F;

    % Displaying results
    displayResults(nodes, elements, T);
end

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

function [K, F] = applyBoundaryConditions(K, F, nodes, elements, convectionCoefficient, ambientTemperature, internalTemperature)
    % Identifying boundary nodes
    internalRadius = 0.01; % Internal radius (m)
    externalRadius = 0.04; % External radius (m)
    
    internal_nodes = find(sqrt(nodes(:,2).^2 + nodes(:,3).^2) < (internalRadius + 0.001));
    external_nodes = find(sqrt(nodes(:,2).^2 + nodes(:,3).^2) > (externalRadius - 0.001));
    
    % Applying fixed temperature boundary condition at internal boundary
    K(internal_nodes, :) = 0;
    K(internal_nodes, internal_nodes) = eye(length(internal_nodes));
    F(internal_nodes) = internalTemperature;
    
    % Applying convective heat transfer boundary condition at external boundary
    for e = 1:size(elements, 1)
        nodeNumbers = elements(e, 2:end);
        if any(ismember(nodeNumbers, external_nodes))
            x = nodes(nodeNumbers, 2);
            y = nodes(nodeNumbers, 3);
            [Ke_convection, Fe_convection] = convectionBoundary(x, y, convectionCoefficient, ambientTemperature);
            K(nodeNumbers, nodeNumbers) = K(nodeNumbers, nodeNumbers) + Ke_convection;
            F(nodeNumbers) = F(nodeNumbers) + Fe_convection;
        end
    end
end

function [Ke_convection, Fe_convection] = convectionBoundary(x, y, convectionCoefficient, ambientTemperature)
    % Calculating edge length (assuming only one edge of the element is on the boundary)
    L = sqrt((x(2)-x(1))^2 + (y(2)-y(1))^2);
    
    % Convection heat transfer stiffness matrix
    Ke_convection = (convectionCoefficient * L / 6) * [2 1 0 0; 1 2 0 0; 0 0 0 0; 0 0 0 0];
    
    % Convection heat transfer force vector
    Fe_convection = (convectionCoefficient * ambientTemperature * L / 2) * [1; 1; 0; 0];
end

function displayResults(nodes, elements, T)
    % Visualize the results of the FEM simulation
    figure;
    patch('Faces', elements(:,2:end), 'Vertices', nodes(:,2:3), 'FaceVertexCData', T, 'FaceColor', 'interp', 'EdgeColor', 'none');
    colorbar;
    title('Temperature Distribution');
    xlabel('X [m]');
    ylabel('Y [m]');
    axis equal;
    colormap(jet);
end

function [nodes, elements] = readMeshFile(fileName)
    % Reads the mesh file and extracts node and element data
    fid = fopen(fileName, 'r');
    nodes = [];
    elements = [];
    line = fgetl(fid);
    while ischar(line)
        if contains(line, '*Node')
            line = fgetl(fid);
            while ~startsWith(line, '*')
                nodeData = sscanf(line, '%f,%f,%f');
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
        else
            line = fgetl(fid);
        end
    end
    fclose(fid);
end
