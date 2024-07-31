function runFEMSimulation()
    clc;
    clear;
    close all;
    x = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, ...
         0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, ...
         0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024, 0.025, ...
         0.027, 0.030, 0.035, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.1];
    y = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.012, 0.015, ...
         0.016, 0.017, 0.018, 0.019, 0.02, 0.022, 0.025, 0.027, 0.030, 0.032, ...
         0.035, 0.037, 0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.1, 0.12];
    [X, Y, NL, NDP, VAL] = generateMeshgridAndBoundaryConditions(x, y);
    [~, result] = calculateVoltage(NL, X, Y, NDP, VAL);
    visualizeResults(result, x, y);
end

function [X, Y, NL, NDP, VAL] = generateMeshgridAndBoundaryConditions(x, y)
    Nodes = zeros(length(y), length(x));
    XX = zeros(1, length(y) * length(x));
    YY = zeros(1, length(y) * length(x));
    NDP = zeros(1, length(y) * length(x));
    VAL = zeros(1, length(y) * length(x));
    n = 0;
    m = 0;
    for j = 1:length(y)
        for i = 1:length(x)
            if (x(i) < 0.01) && ((y(j) < 0.015) && (y(j) > 0.005))
                continue;
            end
            n = n + 1;
            Nodes(j, i) = n;
            XX(n) = x(i);
            YY(n) = y(j);
            if (y(j) == 0) || (x(i) == 0.1) || (y(j) == 0.12)
                m = m + 1;
                NDP(m) = n;
                VAL(m) = 0;
            elseif ((y(j) == 0.005) && (x(i) <= 0.01)) || ((y(j) == 0.015) && (x(i) <= 0.01))
                m = m + 1;
                NDP(m) = n;
                VAL(m) = 220;
            elseif (x(i) == 0.01) && ((y(j) < 0.015) && (y(j) > 0.005))
                m = m + 1;
                NDP(m) = n;
                VAL(m) = 220;
            end
        end
    end
    X = XX(1:n);
    Y = YY(1:n);
    NDP = NDP(1:m);
    VAL = VAL(1:m);
    [r, c] = size(Nodes);
    I = 0;
    NN = zeros((r-1)*(c-1)*2, 3);
    for i = 1:r-1
        for j = 1:c-1
            I = I + 2;
            NN(I, :) = [Nodes(i, j), Nodes(i, j + 1), Nodes(i + 1, j)];
            NN(I + 1, :) = [Nodes(i, j + 1), Nodes(i + 1, j + 1), Nodes(i + 1, j)];
        end
    end
    NL = NN(~any(NN == 0, 2), :);
end

function [V, result] = calculateVoltage(NL, X, Y, NDP, VAL)
    NE = size(NL, 1);    
    ND = length(X);      
    NP = length(NDP);    
    B = zeros(ND, 1);
    C = zeros(ND, ND);
    for I = 1:NE
        K = NL(I, :);
        XL = X(K);
        YL = Y(K);
        P = [YL(2) - YL(3); YL(3) - YL(1); YL(1) - YL(2)];
        Q = [XL(3) - XL(2); XL(1) - XL(3); XL(2) - XL(1)];
        AREA = 0.5 * abs(P(2) * Q(3) - Q(2) * P(3));
        CE = (P * P' + Q * Q') / (4.0 * (AREA + eps));
        for J = 1:3
            IR = K(J);
            IFLAG1 = false;
            if any(IR == NDP)
                C(IR, IR) = 1.0;
                B(IR) = VAL(NDP == IR);
                IFLAG1 = true;
            end
            if ~IFLAG1
                for L = 1:3
                    IC = K(L);
                    if any(IC == NDP)
                        B(IR) = B(IR) - CE(J, L) * VAL(NDP == IC);
                    else
                        C(IR, IC) = C(IR, IC) + CE(J, L);
                    end
                end
            end
        end
    end
    if rank(C) < ND
        error('The global stiffness matrix is singular. Check boundary conditions and mesh.');
    end
    V = C \ B;
    N = (1:ND)';
    result = [N, X', Y', V];
end

function visualizeResults(result, x, y)
    X220 = result(result(:, 4) == 220, 2);
    Y220 = result(result(:, 4) == 220, 3);
    X0 = result(result(:, 4) == 0, 2);
    Y0 = result(result(:, 4) == 0, 3);
    
    [X220, Y220] = removeZeroCoordinates(X220, Y220);
    [X0, Y0] = removeZeroCoordinates(X0, Y0);
    
    [xx, yy] = meshgrid(x, y);
    v = griddata(result(:, 2), result(:, 3), result(:, 4), xx, yy);
    
    figure;
    if numel(X220) == numel(Y220) && ~isempty(X220)
        plot(X220, Y220, 'r', 'linewidth', 5);
        hold on;
    end
    if numel(X0) == numel(Y0) && ~isempty(X0)
        plot(X0, Y0, 'b', 'linewidth', 5);
        hold on;
    end
    
    contourf(xx, yy, v, 20, 'LineColor', 'none');
    colorbar;
    xlabel('X [m]');
    ylabel('Y [m]');
    title('Temperature Distribution | <<Fazel>>');
    hold off;
end

function [X, Y] = removeZeroCoordinates(X, Y)
    X(X == 0) = [];
    Y(Y == 0) = [];
end
%Fazel Mohammad Ali Pour - 7/15/2024 - Matlab Code
