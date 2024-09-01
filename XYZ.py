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

def check_Friedel(hkls):
    if sum((hkls[0] + hkls[1] + hkls[2] + hkls[3]) != [0,0,0]) != 0:
        raise TypeError("Reflections given are not Friedel ones (two by two)")
    if np.sum(hkls[0] + hkls[1]) == 0:
        idxs = [0,1,2,3]
    elif np.sum(hkls[0] + hkls[2]) == 0:
        idxs = [0,2,1,3]
    elif np.sum(hkls[0] + hkls[3]) == 0:
        idxs = [0,3,1,2]
    else:
        raise TypeError("Reflections given are not Friedel ones (two by two)")
    return idxs

def loop(hkls,I,idxs):
    for i in range(len(hkls)):
        if i in idxs:
            continue
        for j in range(1,len(hkls)):
            if j in idxs:
                continue
            if sum(hkls[i] + hkls[j] == [0,0,0]) == 3:
                return i, j
    raise TypeError("{} Friedel pair(s) found, not able to find {}. pair".format(I, I+1))

def find_Friedel(hkls):
    session.log.info("Please give the algorithm a couple of seconds")
    idxs = []
    for I in range(2):
        i, j = loop(hkls, I, idxs)
        idxs.append(i)
        idxs.append(j)
    return idxs


def XYZ(*args):
    indexes = [*args]
    omega0 = 0 #TODO What does this do? Is it offset?
    Dist = detdist.target 
    Dsup = 0 #TODO What is this value?

    # Get sample, instrument, reflection list and motors
    sample = session.experiment.sample
    instr = session.instrument
    rfl = sample.getRefList()
    motors = instr.get_motors()

    # Get Reflections
    hkls = []
    obs = []
    for r in rfl.generate(0):
        hkl_temp = []
        for q in r[0]:
            hkl_temp.append(q)
        hkls.append(hkl_temp)
        i = 0
        obs_temp = []
        for a in r[1]:
            if i < 3:
                obs_temp.append(a)
            i += 1
        obs.append(obs_temp)
    hkls = np.array(hkls)
    obs = np.array(obs)   

    if len(indexes) == 4:
        # Seperate specified indicies
        hkls = np.array([hkls[x] for x in indexes])
        obs = np.array([obs[x] for x in indexes])
        idxs = check_Friedel(hkls)
        hkls = hkls[idxs]
        obs = obs[idxs]

    else:
        session.log.warning("You did not give 4 reflections, attempting to find self")
        idxs = find_Friedel(hkls)
        hkls = hkls[idxs]
        obs = obs[idxs]


    # Print the selected reflections
    session.log.info("")
    session.log.info(' H   K   L   {:^8}  {:^8}  {:^8}'.format(motors[0].name, motors[1].name, motors[2].name))
    for i in range(4):
        hkl = hkls[i]
        ob = obs[i]
        session.log.info('{:>2d}  {:>2d}  {:>2d}  {:>8.4f} {:>8.4f} {:>8.4f}'.format(int(hkl[0]),int(hkl[1]),int(hkl[2]), float(ob[0]),float(ob[1]),float(ob[2])))

    # Z correction
    nu_mil = np.average(obs[:,2])
    DZ = (Dist+Dsup)*np.sin(nu_mil*np.pi/180)
    
    alpha = []
    beta =  []
    delta = []

    # X & Y correction
    for i in range(2):
        alpha.append(np.sin((np.pi/180)*(obs[2*i,1]-omega0-(obs[2*i,0]+obs[2*i+1,0])/2)))
        beta.append(np.cos((np.pi/180)*(obs[2*i,1]-omega0-(obs[2*i,0]+obs[2*i+1,0])/2)))
        delta.append((Dist+Dsup)*np.sin(((obs[2*i,0]-obs[2*i+1,0])/2)*np.pi/180))

    DX =  (delta[0]*beta[1] - delta[1]*beta[0])/ (alpha[0]*beta[1] - alpha[1]*beta[0])
    DY = -(delta[0]*alpha[1] - delta[1]*alpha[0])/ (alpha[0]*beta[1] - alpha[1]*beta[0])
    

    session.log.info(" ")
    session.log.info("Translations to make:")
    session.log.info("---------------------")
    session.log.info("x = {:>8.5f} mm, y = {:>8.5f} mm, z = {:>8.5f} mm".format(-DX, -DY, -DZ))
    session.log.info("---------------------")

    gam_new = [(obs[0,0]+obs[1,0])/2,(obs[0,0]+obs[1,0])/2,(obs[2,0]+obs[3,0])/2,(obs[2,0]+obs[3,0])/2]

    session.log.info(" ")
    session.log.info("New Reflections:")
    session.log.info(' H   K   L   {:^8}  {:^8}  {:^8}'.format(motors[0].name, motors[1].name, motors[2].name))
    for i in range(4):
        hkl = hkls[i]
        ob = obs[i]
        session.log.info('{:>2d}  {:>2d}  {:>2d}  {:>8.4f} {:>8.4f} {:>8.4f}'.format(int(hkl[0]),int(hkl[1]),int(hkl[2]), float(gam_new[i]),float(ob[1]),float(ob[2]-nu_mil)))

 
XYZ()