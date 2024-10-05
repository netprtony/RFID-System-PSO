from random import random
import math

def fitFunc(xVals):
    fitness = 10*len(xVals)
    for i in range(len(xVals)):
        fitness += xVals[i]**2 - (10*math.cos(2*math.pi*xVals[i]))
    return fitness

def initPosition(Np, Nd, xMin, xMax):
    R = [[xMin + random()*(xMax-xMin) for i in range(0, Nd)] for p in range(0, Np)]

    return R

def updatePosition(R, Np, Nd, xMin, xMax):

     for p in range(0, Np):
        for i in range(0, Nd):

            R[p][i] = R[p][i] + V[p][i]

            if R[p][i] > xMax: R[p][i] = xMax
            if R[p][i] < xMin: R[p][i] = xMin

def initVelocity(Np, Nd, vMin, vMax):
    V =[[vMin + random()*(vMax-vMin) for i in range(0,Nd)] for p in range(0, Np)]

    return V

def updateVelocity(R, V, Np, Nd, j, c1, c2, w, vMin, vMax, chi, pBestPos, gBestPos):
    
    for p in range(0, Np):
        for i in range(0, Nd):

            r1 = random()
            r2 = random()

            V[p][i] = chi* (w * V[p][i] + r1*c1*(pBestPos[p][i]-R[p][i]) 
                                        + r2*c2*(gBestPos[i]   -R[p][i]))

            if V[p][i] > vMax: V[p][i] = vMax
            if V[p][i] < vMin: V[p][i] = vMin

def updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestVal):

    for p in range(0, Np):
        M[p] = fitFunc(R[p])

        if M[p] < gBestVal:
            gBestVal = M[p]
            gBestPos   = R[p]

        if M[p] < pBestVal[p]:
            pBestVal[p] = M[p]
            pBestPos[p]   = R[p]

    return gBestVal

if __name__ == "__main__":

    Np, Nd, Nt    = 50, 20, 100
    c1, c2        = 2.05, 2.05
    w, wMin, wMax = 0.0, 0.4, 0.9 

    phi = c1+c2
    chi = 2.0/abs(2.0-phi-math.sqrt(pow(phi, 2)-4*phi))

    xMin, xMax = -5.12, 5.12
    vMin, vMax = 0.25*xMin, 0.25*xMax
    
    gBestValue = float("inf")
    pBestValue = [float("inf")] * Np

    pBestPos   = [[0]*Nd] * Np
    gBestPos   = [0] * Nd

    history    = []
    
    R = initPosition(Np, Nd, xMin, xMax)
    V = initVelocity(Np, Nd, vMin, vMax)
    M = [fitFunc(R[p]) for p in range(0, Np)]

    for j in range(0, Nt):

        updatePosition(R, Np, Nd, xMin, xMax)

        gBestValue = updateFitness(R, M, Np, pBestPos, pBestValue, gBestPos, gBestValue)
        history.append(gBestValue)
        
        w = wMax - ((wMax-wMin)/Nt)*j
        updateVelocity(R, V, Np, Nd, j, c1, c2, w, vMin, vMax, chi, pBestPos, gBestPos)

    for h in history:
        print(h)