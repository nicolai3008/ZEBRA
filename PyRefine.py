# *****************************************************************************
# NICOS, the Networked Instrument Control System of the MLZ
# Copyright (c) 2009-2024 by the NICOS contributors (see AUTHORS)
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Module authors:
#   Nicolai Amin <nicolai.amin@psi.ch>
#
# *****************************************************************************

from nicos import session
from nicos_sinq.sxtal.singlexlib import calculateBMatrix as genB
from nicos_sinq.sxtal.singlexlib import matFromTwoVectors
import numpy as np
import os
from os import path
from scipy.optimize import least_squares
from scipy import linalg, optimize

# Normalize vector
def normv(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return 0, v
    return norm,  v / norm


# The matrices below are different from the rotations in singlexlib
def Xrot(a, typecode='d'):
    """Construct a rotation matrix rotating 'a' around the X axis"""
    a *= np.pi/180
    sa = np.sin(a)
    ca = np.cos(a)
    return np.array([[1.0, 0, 0], [0, ca, sa], [0, -sa, ca]], typecode)
def Yrot(a, typecode='d'):
    """Construct a rotation matrix rotating 'a' around the Y axis"""
    a *= np.pi/180
    sa = np.sin(a)
    ca = np.cos(a)
    return np.array([[ca, 0, -sa], [0, 1.0, 0], [sa, 0, ca]], typecode)
def Zrot(a, typecode='d'):
    """Construct a rotation matrix rotating 'a' around the Z axis"""
    a *= np.pi/180
    sa = np.sin(a)
    ca = np.cos(a)
    return np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1.0]],
                    typecode)
def Rot(r, typecode='d'):
    """Construction of full rotation matrix"""
    X = Xrot(r[0])
    Y = Yrot(r[1])
    Z = Zrot(r[2])
    return Z @ Y @ Z



def couple(hkls, obs, instr, sample, min, refs = None):
    """"
    Couple function to find the optimal two reflections to run PyRefine on
    """
    PHIF = 0
    V3MX = 0
    idxs = [-1,-1]
    cell = sample.getCell()
    B = genB(cell)
    if refs != None:
        hkls = [hkls[i] for i in refs]
    for i in range(len(hkls)-1):        
        for j in range(i+1, len(hkls)):
            h1 = hkls[i]
            V1 = np.matmul(B,h1)
            h2 = hkls[j]
            V2 = np.matmul(B,h2)
            V3 = np.cross(V1, V2)
            V1M, V1 = normv(V1)
            V2M, V2 = normv(V2)
            V3M, V3 = normv(V3)
            V12M = V1M * V2M

            if abs(V12M) < 10**(-5):
                continue
            if abs(V3M/V12M) <= 1:
                PHI = np.arcsin(V3M/V12M)
            else:
                continue

            PHI = np.abs(PHI % np.pi)

            if np.pi/2 <= PHI:
                np.pi - abs(PHI)

            if PHI < min*np.pi/180:
                continue

            if ((45 < abs(obs[i,2])) and (abs(obs[i,2]) < 135) and (45 < abs(obs[j,2])) and (abs(obs[j,2]) < 135)):
                continue

            if V3M < V3MX:
                continue

            PHIF = PHI
            V3MX = V3M
            idxs = [i, j]

    if idxs == [-1,-1]:
        if refs != None: 
            session.log.error('Two selected reflections are less than %i degrees seperated \n Please choose others',min)
            raise TypeError()
        else: 
            session.log.error('No two reflections are more than %i degrees seperated \n Please measure more Peaks',min)
            raise TypeError()

    else:
        session.log.info('Orienting Ref 1. %i  %i %i %i', idxs[0], hkls[idxs[0],0],hkls[idxs[0],1],hkls[idxs[0],2])
        session.log.info('Orienting Ref 2. %i  %i %i %i', idxs[1], hkls[idxs[1],0],hkls[idxs[1],1],hkls[idxs[1],2])
        session.log.info('Angle: %f', PHIF*180/np.pi)
        return (idxs[0], idxs[1])


def UNBVectorFromAngles(reflection):
    """"
    Calculate the B&L U vector from normal beam geometry angles
    Not using normal function as it uses wrong references
    """
    u = np.zeros((3,), dtype='float64')
    gamma = np.deg2rad(reflection[0])
    om = np.deg2rad(reflection[1])
    nu = np.deg2rad(reflection[2])
    u[0] = sin(gamma) * cos(om) * cos(nu) + \
        sin(om) * (1. - cos(gamma) * cos(nu))
    u[1] = sin(gamma) * sin(om) * cos(nu) - \
        cos(om) * (1. - cos(gamma) * cos(nu))
    u[2] = sin(nu)
    return u

def UVectorFromAngles(reflection):
    """
    Calculate the B&L U vector from normal beam geometry angles
    Not using normal function as it uses wrong references
    """
    u = np.zeros((3,), dtype='float64')

    # The tricky bit is set again: Busing & Levy's omega is 0 in
    # bisecting position. This is why we have to correct for
    # stt/2 here
    om = np.deg2rad(reflection[1] - reflection[0]/2)
    chi = np.deg2rad(reflection[2])
    phi = np.deg2rad(reflection[3])
    u[0] = cos(om) * cos(chi) * cos(phi) - sin(om) * sin(phi)
    u[1] = cos(om) * cos(chi) * sin(phi) + sin(om) * cos(phi)
    u[2] = cos(om) * sin(chi)
    return u

def UKVectorFromAngles(reflection):
    """
    Calculate the B&L U vector from normal beam geometry angles
    Not using normal function as it uses wrong references
    """
    
    return 0


def vecori(v1, v2, dials):
    """
    Function to rotate calculated orienting reflections such 
    that they are in plane with the actual reflections
    """
    RAD2DEG = 180/np.pi
    dials = np.array(dials) / RAD2DEG 
    VN = np.cross(v1,v2)
    VXM, VX = normv(v1)
    VZM, VZ = normv(VN)
    VYM, VY = normv(np.cross(VZ,VX))

    for i in range(3):
        j = (1+i) % 3
        k = (2+i) % 3
        B = np.zeros((3,3))
        B[:,i] = VX
        B[:,j] = VY
        B[:,k] = VZ
        BI = B.T
        V01 = BI @ v1
        V02 = BI @ v2
        CE = 1 - dials[i]**2 /2
        SE = dials[i]
        C = np.array([[1,0,0],[0,CE,-SE],[0,SE,CE]])
        V001 = C @ V01
        V002 = C @ V02
        v1 = B @ V001
        v2 = B @ V002
    return v1, v2


def tr1p(X,Y):
    """
    Function to create a vector space
    """ 
    # Normalize X
    XM, XX = normv(X)

    # Calculate Z
    Z = np.cross(X,Y)
    ZM, ZZ = normv(Z)

    # Calculate cross product of Z and X
    YY = np.cross(ZZ, XX)
    XYZ = np.column_stack((XX, YY, ZZ))

    return XYZ


# Function to calculate and set a UB matrix
def newUB(instr, sample, rfl, offsets, disps, dials, refine_type):
    """
    Function to calculate UB matrix from two orienting reflections
    args:
        instr   - instrument that this is run on
        sample  - sample that is being tested
        rfl     - list of reflections that have been measured
        offsets - motor offsets for the reflections
        dials   - rotation of the UB matrix. Only used in the optimization.
    """
    r1_temp = rfl.get_reflection(instr.orienting_reflections[0])
    r2_temp = rfl.get_reflection(instr.orienting_reflections[1])
    r11 = (r1_temp[0])
    r21 = (r2_temp[0])
    r12 = tuple([r1_temp[1][i]-offsets[i]+displacements(refine_type, r1_temp[1][i]-offsets[i], disps, i) for i in range(len(offsets))])
    r22 = tuple([r2_temp[1][i]-offsets[i]+displacements(refine_type, r2_temp[1][i]-offsets[i], disps, i) for i in range(len(offsets))])
    r1 = (r11,r12,())
    r2 = (r21,r22,())
    B = genB(sample.getCell())
    if not B.any():
        return None
    h1 = B.dot(r11)
    h2 = B.dot(r21)
    HT = matFromTwoVectors(h1, h2)
    if refine_type == "NB":
        u1 = UNBVectorFromAngles(r12)
        u2 = UNBVectorFromAngles(r22)
    elif refine_type == "Euler":
        u1 = UVectorFromAngles(r12)
        u2 = UVectorFromAngles(r22)
    elif refine_type == "Kappa":
        u1 = UKVectorFromAngles(r12)
        u2 = UKVectorFromAngles(r22)
    UT = matFromTwoVectors(u1, u2)
    HTT = HT.transpose()
    U = UT.dot(HTT)
    UB = U.dot(B)
    v1, v2 = UB.dot(r11) , UB.dot(r21)
    v1, v2 = vecori(v1, v2, dials)
    TR1PC = tr1p(h1, h2)
    TR1PC = TR1PC.T
    TR1PO = tr1p(v1, v2)
    U = np.dot(TR1PO,TR1PC)
    sample.ubmatrix = list(np.dot(U,B).flatten())


def displacements(refine_type, angle, disps, index):
    DEG2RAD = np.pi/180
    D = detdist.target
    disp = 0
    if refine_type == "NB":
        if index == 2:
            disp += np.arcsin(np.sin(angle*DEG2RAD)*disps[2]/D)/DEG2RAD
        if index == 1:
            disp += np.arcsin(np.sin(angle*DEG2RAD+np.arctan2(disps[0],disps[1]))*np.sqrt(disps[1]**2+disps[0]**2)/D)/DEG2RAD
    return disp


# Residual function for optimization. Calculates positions of reflections
# and finds the difference between the observed values and calculated 
# values. 
def residuals(param, sample, instr, limits, rfl, hkls, obs, refine_type, verbose):
    """
    Function that calculates the total absolute difference between
    calculated and observed reflections.
    args;
        param   - parameters that can be varied
                  set up to be [cell, motor offset, dials]
        instr   - instrument that this is run on
        sample  - sample that is being tested
        limits  - limits to make sure that coupled values are coupled
        rfl     - list of reflections that have been measured
        hkl     - hkl reflections to compare to
        obs     - observed angles to compare to
        refine_type - which refinement NB, Euler, Kappa
    """
    # If there are coupled parameters, this will set them to be the same value
    # it is coupled to.
    for i, lim in enumerate(limits):
        if lim > 1:
            con = lim-2
            param[i] = param[con]

    # Set cell paramters to calculate UB matrix correctly
    sample.a = param[0]
    sample.b = param[1]
    sample.c = param[2]
    sample.alpha = param[3]
    sample.beta = param[4]
    sample.gamma = param[5]

    # Split up offsets and dials
    offsets = np.array(param[6:len(param)-6])
    disps = param[-6:-3]
    dials = param[-3:]

    # Calculate new UB matrix
    newUB(instr,sample, rfl, offsets, disps, dials, refine_type)
    

    # Lists to fill up with data. Calc is not used at the moment
    # but can be used as a debugging tool to compare to fortran rafin
    calc = []
    diff = []

    # Loop through reflections
    for i, hkl in enumerate(hkls):
        # Calculate positions of angles
        poslist = np.array(instr._extractPos(instr._calcPos(hkl)))
        diff_temp = []
        calc_temp = []
        for j, ang in enumerate(poslist):
            if j > 2:
                continue
            a = float(ang[1]) + offsets[j]
            a += displacements(refine_type, a, disps, j)

            # Calculate difference
            if a > 180:
                calc_temp.append(a-360)
            elif a < -180:
                calc_temp.append(a+360)
            else:
                calc_temp.append(a)
            diff_temp.append((obs[i,j]-a + 180) % 360 - 180)

        diff.append(diff_temp)
        calc.append(calc_temp)

    # Return cost function
    calc = np.array(calc)
    diff = np.array(diff)

    if verbose == 1:
        session.log.info('------------------------------------------------')
        motors = instr.get_motors()
        session.log.info(' H   K   L   {:^27}  {:^29}  {:^29}'.format(motors[0].name, motors[1].name, motors[2].name))
        session.log.info('          {:^8} {:^8} {:^9}  {:^9} {:^9} {:^9}  {:^9} {:^9} {:^9}'.format('obs', 'calc', 'diff','obs', 'calc', 'diff','obs', 'calc', 'diff'))
        for i, hkl in enumerate(hkls):
            session.log.info('{:>2d}  {:>2d}  {:>2d}  {:>8.4f} {:>8.4f} {:>9f}  {:>9.4f} {:>9.4f} {:>9f}  {:>9.4f} {:>9.4f} {:>9f}'.format(int(hkl[0]),int(hkl[1]),int(hkl[2]), obs[i,0], calc[i,0], diff[i,0], obs[i,1], calc[i,1], diff[i,1],obs[i,2], calc[i,2], diff[i,2]))
        session.log.info('------------------------------------------------')
        session.log.info('            {:^8} {:^8} {:>9f}  {:^9} {:^9} {:>9f}  {:^9} {:^9} {:>9f}'.format('', 'Mean dev', np.average(np.abs(diff[:,0])),'', 'Mean dev', np.average(np.abs(diff[:,1])),'', 'Mean dev', np.average(np.abs(diff[:,2]))))
    return calc, diff


# Function to set the initial guess and bounds for the least squares method
def p0_bounds(limits, instr, sample):
    """
    Function to define the initial guess for the least squares method
    as well as the boundaries which the method has to keep the the parameters
    args;
        limits  - limits set by the user to define which variables should change
                  and which should not. List of binaries
        instr   - instrument that this is run on
        sample  - sample that is being tested
    """
    # The least-squares method requires the parameters to have some wiggle room
    # and a good approximation for 0 is 10^-10 :)
    eps=1e-10

    # Get unit cell parameters
    cell = sample.getCell()
    cell_init = [cell.a,cell.b,cell.c,cell.alpha,cell.beta,cell.gamma]

    # Get amount of motors
    motors = instr.get_motors()
    init_offsets = [mot.offset for mot in motors]

    # Set initial sample displacement
    disp_array = [0,0,0]

    # Set initial dial rotations
    rotation_array = [0,0,0]

    # Collect them all
    parameters = cell_init + init_offsets + disp_array + rotation_array
    
# Setting bounds for each parameter depending on limits input
    lower = []
    upper = []
    for i in range(len(parameters)):
        # Limits for unit cell lengths
        if i < len(cell_init)/2:
            if limits[i] != 0:
                lower.append(0)
                upper.append(np.inf)
            else:
                lower.append(parameters[i])
                upper.append(parameters[i]+eps)
        
        # Limits for unit cell angles
        elif i < len(cell_init):
            if limits[i] != 0:
                lower.append(0)
                upper.append(180)
            else:
                lower.append(parameters[i])
                upper.append(parameters[i]+eps)
        
        # Limits for motor offsets
        elif i < len(cell_init)+len(init_offsets):
            if limits[i] == 1:
                lower.append(-180)
                upper.append(180)
            else:
                lower.append(parameters[i])
                upper.append(parameters[i]+eps)

        # limits for sample displacement
        elif i < len(cell_init)+len(init_offsets):
            if limits[i] == 1:
                lower.append(-20)
                upper.append(20)
            else:
                lower.append(parameters[i])
                upper.append(parameters[i]+eps)
        # Limits for dials
        else:
            lower.append(-180)
            upper.append(180)

    return parameters, (lower,upper)


def SID(X):
    """
    Safe invert and determinant of matrix
    """
    n = X.shape[1] - 1
    AMAT = X[:,:n]
    VVEC = X[:,n]
    AMAT_inv = np.linalg.inv(AMAT)
    D = np.linalg.det(AMAT)
    solution = np.dot(AMAT_inv,VVEC)
    A_inv_aug = np.column_stack((AMAT_inv,solution))
    return A_inv_aug, D


def refinement(func, p0, sample, instr, limits, rfl, hkls, obs, bounds, refine_type, max_nfev=5, eps = 0.001, verbose=0):
    """ 
    Function to refine the parameters. Algorithm is the same as that of Rafin
    Should nearly always converge
    args;
        verbose     - (0/1/2) How much should the function print
        replace     - (0/1) Should the function replace the cell paramters
                            and UB matrix
        max_nfev    - Amount of cycles (5)
        eps         - Magnitude of finit increase (0.001)
    """
    # Radians to Degrees
    RAD2DEG = 180/np.pi
    verb = int(verbose > 1)
    # Find which parameters to change
    p_idxs = np.where(np.array(limits)==1)[0]

    # Matrix sizes
    NP = len(p_idxs)
    NOL = len(hkls)

    # Print information from initial cycle
    session.log.info('')
    session.log.info('')
    session.log.info('Cycle Init')
    calc0, c0 = residuals(p0, sample, instr, limits, rfl, hkls, obs, refine_type, 1)

    # Cost and deviation arrays
    cost = []
    dev = []
    for i in range(max_nfev):
        if verbose > 0:
            session.log.info('')
            session.log.info('')
            session.log.info('Cycle {}'.format(i))

        # Calculated reflections and cost for initial paramters
        calc0, c0 = residuals(p0, sample, instr, limits, rfl, hkls, obs, refine_type, verb)
        cost.append(sum(((c0/RAD2DEG).flatten())**2))
        dev.append(np.sqrt(cost[i]/(3*NOL)))
        DYC = np.zeros((NP,NOL,3))
        VVEC = np.zeros((NP,1))
        AMAT = np.zeros((NP,NP))
        DELP = np.zeros(NP)
        
        # Slightly vary parameters and calculate differences
        for j in range(NP):
            p0[p_idxs[j]] += eps
            calc, c = residuals(p0, sample, instr, limits, rfl, hkls, calc0, refine_type, 1)
            p0[p_idxs[j]] -= eps
            DYC[j] = -c/eps/RAD2DEG
      
        # Calculate two structures:
        # VVEC - The gradient of cost multiplied by the cost for each change
        # AMAT - Matrix of how each parameter change affects each other
        #
        # The specifics here is to find a pseudo-gradient for the system and then
        # minimize with respect to it by finding the inverse and solving the system
        for I in range(NP):
            VVEC[I] = np.sum(np.multiply(DYC[I],c0/RAD2DEG))
            for J in range(NP):
                AMAT[I,J] = np.sum(np.multiply(DYC[I],DYC[J]))
        X = np.hstack((AMAT,VVEC))

        # Find the inverse and determinant
        X, D = SID(X)
        if verbose > 0:
            session.log.info("Sums: {:5f},Std.Dev.: {:5f},Deter.: {:5e}".format(cost[i],dev[i],D))
        PDELP = 0

        # DELP is the amount that each parameter should change by
        # PDELP is correlated to the variance in the parameter
        for I in range(NP):
            DELP[I] = X[I,-1]
            PDELP += DELP[I]*VVEC[I]
            p0[p_idxs[I]] += X[I,-1]

        # Print change in parameter
        if verbose > 0:
            pr = ' '.join(['{:8.5f}'.format(i) for i in DELP])
            session.log.info("Change in P: "+ pr)
            pr = ' '.join(['{:8.5f}'.format(i) for i in np.array(p0)[p_idxs]])
            session.log.info("New Paras:   "+ pr)

        # Convergence possibility either cost is not lowering or the parameters aren't
        if i > 2:
            if (cost[i-1]/cost[i-1] - 1 <= 10**(-6)):
                session.log.info("Converged with respect to residuals")
                break
        if any(np.abs(DELP) < 10**(-6)):
            session.log.info("Converged with respect to parameters")
            break


    # Log Final cycle, costs and return the new values, vairances and correlations
    session.log.info('')
    session.log.info('')
    session.log.info('Cycle Final')
    calc0, c0 = residuals(p0, sample, instr, limits, rfl, hkls, obs, refine_type, 1)
    cost.append(sum(c0.flatten()**2))
    var = np.zeros(NP)
    cor = np.zeros((NP,NP))
    FT = 3*NOL-NP
    for i in range(NP):
        var[i] = AMAT[i,i]*(cost[-1]-PDELP)/FT
        for j in range(NP):
            cor[i,j] = AMAT[i,j]/sqrt(AMAT[i,i]*AMAT[j,j])
    return cost, p0, np.sqrt(var), cor  

   
# Refinement method
def PyRefine(*args, orienting_refs=None, ignore_refs=[], verbose=1, replace=False, min_angle=45, ncyc=5, eps = 0.001):
    """ 
    Function to define the initial guess for the least squares method
    as well as the boundaries which the method has to keep the the parameters.
    When running, input should be a list of any type, with the parameters that should
    be varied. If variables should be dependant on other, then the shape of the list
    should be fx. (a,[b,c]), where b and c now are forced to be the same value as a.
    args;
        *args            - Which parameters should be varied (a, b, c, etc.) Will give warning
                           if input is not a parameter to vary
                           Possible parameters: a, b, c, alpha, beta, gamma, motor names
                           (stt, som, etc.)
        orienting_refs   - If not None, chooses the two orienting reflections for the UB matrix
                           Still have to be further away than the min_angle
        ignore_refs      - Reflections to ignore
        verbose          - (0/1/2) How much should the function print
        replace          - (0/1) Should the function replace the cell paramters
                                 and UB matrix
        min_angle        - Minimum angle between orienting reflection (45)
        ncyc             - Amount of cycles (10)
        eps              - Magnitude of finit increase
    """
    # Get sample, instrument, reflection list and motors
    sample = session.experiment.sample
    instr = session.instrument
    rfl = sample.getRefList()
    motors = instr.get_motors()
    
    # Get instrument mode
    if type(instr).__name__ is "SinqNB":
        refine_type = "NB"
    elif type(instr).__name__ is "SinqEuler":
        refine_type = "Euler"
    elif type(instr).__name__ is "SinqKappa":
        refine_type = "Kappa"
    else:
        session.log.warning("Instrument mode not found (not NB, Euler or Kappa). Resorting to NB, but please fix")
        refine_type = "NB"

    # For normal beam, omega cannot be optimized. This is due to the fact that a rotation in omega is arbitrary,
    # and will cause no difference in the calculated angles. It will however, completely mess up the inversion of 
    # gradient matrix, causing a singular matrix after enough iterations.
    if ("om" in args) and (refine_type == "NB"):
        session.log.error("It is not possible to optimize omega. Please pick another angle to optimize")
        TypeError()

    # Setting initial parameter guess, as well as which parameters can vary using p0_bounds()
    names = ["a","b",'c','alpha','beta','gamma'] + [mot.name for mot in motors] + ['dx', 'dy', 'dz']
    limits = [0]*len(names)
    for i, arg in enumerate(args):
        if isinstance(arg,list):
            for j in arg:
                if j in names:
                    idx = names.index(j)
                    idx_base = names.index(args[i-1])
                    limits[idx] = 2+idx_base
                else:
                    # Will alert if a variable is set incorrectly
                    session.log.warning('Coupled Variable "%s" does not exist', j)
        else:
            if arg in names:
                idx = names.index(arg)
                limits[idx] = 1
            else:
                # Will alert if a variable is set incorrectly
                session.log.warning('Variable "%s" does not exist', arg)
    names += ["Rotation X", "Rotation Y", "Rotation Z"]
    limits += [1,1,1]
    p0, bounds = p0_bounds(limits, instr, sample)
    p_init = p0.copy()

    # Print before
    session.log.info('*** PyRafin *** version 1.0, N. Amin 16-Aug-2024')
    session.log.info('-------------------- TITLE ---------------------')
    session.log.info(sample.name)
    session.log.info('------------------------------------------------')
    session.log.info('')
    session.log.info('Verbosity: \t\t\t {}'.format(verbose))
    session.log.info('Replace: \t\t\t {}'.format(replace))
    session.log.info('Min angle coupling: \t\t {}'.format(45))
    session.log.info('Max cycles: \t\t\t {}'.format(10))
    session.log.info('')
    session.log.info('Wavelength: \t\t\t {}'.format(instr.wavelength))


    # Print which mode zebra is in
    session.log.info('')
    session.log.info('Will refine {} UB Matrix'.format(refine_type))
    session.log.info('')

    # Print Parameters
    session.log.info('Cell Parameters:')
    for i in range(6):
        n = names[i]
        session.log.info('\t {} = {} \t varying = {}'.format(n, p0[i], limits[i]))
    session.log.info('Motor offsets:')
    for mot in motors:
        n = mot.name
        session.log.info('\t {} = {} \t\t varying = {}'.format(n, p0[names.index(n)], limits[names.index(n)]))
    session.log.info('Sample Displacement:')
    for i in range(6+len(motors),6+len(motors)+3):
        n = names[i]
        session.log.info('\t {} = {} \t\t varying = {}'.format(n, p0[names.index(n)], limits[names.index(n)]))
    session.log.info('')
    session.log.info('')

    # Get information out of reflection list
    hkls = []
    obs = []
    for i, r in enumerate(rfl.generate(0)):
        if i in ignore_refs:
            continue
        hkl_temp = []
        for q in r[0]:
            hkl_temp.append(q)
        hkls.append(hkl_temp)
        obs_temp = []
        for i, a in enumerate(r[1]):
            if i > 2:
                continue
            obs_temp.append(a)
        obs.append(obs_temp)
    hkls = np.array(hkls)
    obs = np.array(obs)

    # Find orienting reflections to look at, calculate UB from there reflections
    if orienting_refs != None:
        if len(orienting_refs) == 2:
            instr.orienting_reflections = couple(hkls, obs, instr, sample, min=min_angle, refs = orienting_refs)
            ListRef()
            newUB(instr, sample, rfl, [0]*len(motors), [0,0,0], [0,0,0], refine_type)
            init_UB = sample.ubmatrix
        else:
            session.log.info('Not two orienting reflections found, finding own')
            instr.orienting_reflections = couple(hkls, obs, instr, sample, min=min_angle)
            newUB(instr, sample, rfl, [0]*len(motors), [0,0,0], [0,0,0], refine_type)
            init_UB = sample.ubmatrix
    else:
        instr.orienting_reflections = couple(hkls, obs, instr, sample, min=min_angle)
        newUB(instr, sample, rfl, [0]*len(motors), [0,0,0], [0,0,0], refine_type)
        init_UB = sample.ubmatrix

    # Rafin optimization model, which uses the residuals function to calculate the 
    # optimal paramters, with initial guess p0. Uses inverse matrix calculations
    # and small steps to figure out which direction to move the parameters in
    session.log.info('')
    cost, p1, perr, cor = refinement(residuals, p0, sample, instr, limits, rfl, hkls, obs, limits, refine_type, ncyc, eps, verbose)

    # Print results
    session.log.info('Parameter Final Values and Correlations:')
    num = 0
    for i in range(len(p0)):
        if limits[i] != 1:
            continue
        n = names[i]
        pr = ' '.join(['{:7.4f}'.format(i) for i in cor[num]])
        session.log.info('\t {:>10} = {:>6.4f} (± {:>6.4f})        '.format(n, p1[i], perr[num])+pr)
        num += 1
    session.log.info('')
    session.log.info("Refined UB matrix.")
    for i in range(3):
        session.log.info(" {:>11.8f}  {:>11.8f}  {:>11.8f}".format(sample.ubmatrix[i*3],sample.ubmatrix[i*3+1],sample.ubmatrix[i*3+2]))
        session.log.info('')

    # Replace parametes (or don't)
    if not replace:
        session.log.info('Not Replacing old values')
        sample.ubmatrix = init_UB
        sample.a = p_init[0]
        sample.b =  p_init[1]
        sample.c =  p_init[2]
        sample.alpha =  p_init[3]
        sample.beta =  p_init[4]
        sample.gamma =  p_init[5]
    session.log.info("Done, Goodbye :)")


