import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#inbound mesh | hard coded
def runFEMSimulation():
    x = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
         0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017,
         0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024, 0.025,
         0.027, 0.030, 0.035, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.1]
    y = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.012, 0.015,
         0.016, 0.017, 0.018, 0.019, 0.02, 0.022, 0.025, 0.027, 0.030, 0.032,
         0.035, 0.037, 0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.1, 0.12]
    
    X, Y, NL, NDP, VAL = generateMeshgridAndBoundaryConditions(x, y)
    _, result = calculateVoltage(NL, X, Y, NDP, VAL)
    visualizeResults(result, x, y)

def generateMeshgridAndBoundaryConditions(x, y):
    Nodes = np.zeros((len(y), len(x)), dtype=int)
    XX = []
    YY = []
    NDP = []
    VAL = []
    n = 0
    m = 0
    
    for j in range(len(y)):
        for i in range(len(x)):
            if (x[i] < 0.01) and ((y[j] < 0.015) and (y[j] > 0.005)):
                continue
            n += 1
            Nodes[j, i] = n
            XX.append(x[i])
            YY.append(y[j])
            if (y[j] == 0) or (x[i] == 0.1) or (y[j] == 0.12):
                m += 1
                NDP.append(n)
                VAL.append(0)
            elif ((y[j] == 0.005) and (x[i] <= 0.01)) or ((y[j] == 0.015) and (x[i] <= 0.01)):
                m += 1
                NDP.append(n)
                VAL.append(220)
            elif (x[i] == 0.01) and ((y[j] < 0.015) and (y[j] > 0.005)):
                m += 1
                NDP.append(n)
                VAL.append(220)
    
    X = np.array(XX)
    Y = np.array(YY)
    NDP = np.array(NDP)
    VAL = np.array(VAL)
    r, c = Nodes.shape
    I = 0
    NN = []
    
    for i in range(r - 1):
        for j in range(c - 1):
            I += 2
            NN.append([Nodes[i, j], Nodes[i, j + 1], Nodes[i + 1, j]])
            NN.append([Nodes[i, j + 1], Nodes[i + 1, j + 1], Nodes[i + 1, j]])
    
    NL = np.array(NN)
    NL = NL[~np.any(NL == 0, axis=1)]
    
    return X, Y, NL, NDP, VAL

def calculateVoltage(NL, X, Y, NDP, VAL):
    NE = NL.shape[0]
    ND = len(X)
    NP = len(NDP)
    B = np.zeros(ND)
    C = np.zeros((ND, ND))
    
    for I in range(NE):
        K = NL[I, :]
        XL = X[K-1]
        YL = Y[K-1]
        P = np.array([YL[1] - YL[2], YL[2] - YL[0], YL[0] - YL[1]])
        Q = np.array([XL[2] - XL[1], XL[0] - XL[2], XL[1] - XL[0]])
        AREA = 0.5 * abs(P[1] * Q[2] - Q[1] * P[2])
        CE = (np.outer(P, P) + np.outer(Q, Q)) / (4.0 * (AREA + np.finfo(float).eps))
        
        for J in range(3):
            IR = K[J] - 1
            IFLAG1 = False
            if IR + 1 in NDP:
                C[IR, IR] = 1.0
                B[IR] = VAL[NDP == IR + 1]
                IFLAG1 = True
            if not IFLAG1:
                for L in range(3):
                    IC = K[L] - 1
                    if IC + 1 in NDP:
                        B[IR] -= CE[J, L] * VAL[NDP == IC + 1]
                    else:
                        C[IR, IC] += CE[J, L]
    
    if np.linalg.matrix_rank(C) < ND:
        raise ValueError('The global stiffness matrix is singular. Check boundary conditions and mesh.')
    
    V = np.linalg.solve(C, B)
    result = np.column_stack((np.arange(1, ND + 1), X, Y, V))
    
    return V, result

def visualizeResults(result, x, y):
    X220 = result[result[:, 3] == 220, 1]
    Y220 = result[result[:, 3] == 220, 2]
    X0 = result[result[:, 3] == 0, 1]
    Y0 = result[result[:, 3] == 0, 2]
    
    X220, Y220 = removeZeroCoordinates(X220, Y220)
    X0, Y0 = removeZeroCoordinates(X0, Y0)
    
    xx, yy = np.meshgrid(x, y)
    v = griddata((result[:, 1], result[:, 2]), result[:, 3], (xx, yy), method='linear')
    
    plt.figure()
    if len(X220) == len(Y220) and len(X220) > 0:
        plt.plot(X220, Y220, 'r', linewidth=5)
    if len(X0) == len(Y0) and len(X0) > 0:
        plt.plot(X0, Y0, 'b', linewidth=5)
    
    plt.contourf(xx, yy, v, 20, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Temperature Distribution | <<Fazel>>')
    plt.show()

def removeZeroCoordinates(X, Y):
    X = X[X != 0]
    Y = Y[Y != 0]
    return X, Y

runFEMSimulation()
#Fazel Mohammad Ali Pour - 7/15/2024 - Python Code For The Pdf