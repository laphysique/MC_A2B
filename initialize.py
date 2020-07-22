# Mar 16, 2019 by caseypie
# - Monte Carlo lattice model algorithm for SynGAP-PSD95 phase separation system
# - Four types of molecules: A, B, AB, A2B

# Initializing a box containing molecules for simulation

# Data of molecules are stored in three components:
# 1. An numpy array SPACE stores the STATE of each lattice site:
#    0 for empty
#    1 for freeA
#    2 for freeB
#    3 for A in AB 
#    4 for B in AB
#    5 for A in A2B
#    6 for B in A2B
#    7 for A in A2B in a cluster movement 
#    8 for B in A2B in a cluster movement 
# 2. Four Python lists A, B, AB, A2B store the REAL-space coordinates 
#    of corresponsing molecules:
#     A:   elements are [xA. yA, zA]
#     B:   elements are [xB. yB, zB]
#     AB:  elements are [[xA,yA,zA],[xB. yB, zB]] 
#     A2B: elements are [[xA1,yA1,zA1],[xB. yB, zB],[xA2.yA2,zA2]]
# 3. A 3D Python list SPACE_REAL tracks all REAL-space coordinates.
#    SPACE_REAL[i][j][k] stores the REAL-space coordinates of the molecule (partially)
#    occupying SPACE[i,j,k]:
#     Empty: the element will be np.nan
#     A:     the element will be [xA,yA,zA]
#     B:     the element will be [xB,yB,zB]
#     AB:    the element will be [[xA,yA,zA], [xB,yB,zB]] 
#     A2B:   the element will be [[xA1,yA1,zA1], [xB,yB,zB], [xA2,yA2,zA2]]  
#    Notice that, for an AB or A2B, SPACE_REAL[i][j][k] will also store coordinate 
#    information of its other components

import numpy as np
import mc_steps as mc
import io_control
import parameter_setting as parset

"""
mc.L   = 40  # MC cubic box edge length
mc.Ub1 = 5.  # binding energy of A+B->AB 
mc.Ub2 = 10. # binding energy of AB+A->A2B
mc.chi  = 2. # phase separation Flory-Huggins parameter between each pair of A-B in two A2B 
mc.A2B_isL = True   # True: A-B-A must be in L-shape conformation
mc.A2B_rod = False  # True: A-B-A must be in linear-shape conformation 
"""

def initialize_the_box( pars=None, inifile=None  ):
    
    # Initialize the system by randomly distributing free A and B molecules
    if inifile == None:
        print( 'Initialize the system by parameters fetched from parameters_setting.py' )
        print('Initial configuration: randomly distributed free A and B molecules')

        if pars==None:
            pars = ( parset.NA, parset.NB, parset.L, \
                     parset.Ub1, parset.Ub2, parset.chi, \
                     parset.isL, parset.rod   )

        mc.NA = pars[0]         # number of free A
        mc.NB = pars[1]         # number of free B  
        mc.L  = pars[2]         # MC cubic box edge length  
        mc.Ub1 = pars[3]        # binding energy of A+B->AB  
        mc.Ub2 = pars[4]        # binding energy of AB+A->A2B
        mc.chi = pars[5]        # phase separation Flory-Huggins parameter 
        mc.A2B_isL = pars[6]    # True: A-B-A must be in L-shape conformation 
        mc.A2B_rod = pars[7]    # True: A-B-A must be in linear-shape conformation  
  
        L, NA, NB = mc.L, mc.NA, mc.NB

        # Randomly generate 3D coordinates for free A and B
        Insert = np.random.choice(L*L*L, NA+NB)
        Xini = np.floor(Insert / (L*L) ).astype(int)
        Yini = np.floor( (Insert-L*L*Xini) / L ).astype(int)
        Zini = np.floor( Insert-L*L*Xini-L*Yini ).astype(int)
        
        # initialize A, B, AB, A2B lists
        mc.A = np.concatenate((Xini[0:NA],Yini[0:NA],Zini[0:NA])).reshape(3,NA).T.tolist()
        mc.B = np.concatenate((Xini[NA:],Yini[NA:],Zini[NA:])).reshape(3,NB).T.tolist()
        mc.AB = []  # coordinates of [A,B]
        mc.A2B = [] # coordinates of [A,B,A]

        # initialize SPACE
        mc.SPACE = np.zeros((L,L,L))
        mc.SPACE[Xini[0:NA],Yini[0:NA],Zini[0:NA]] = 1 # type freeA
        mc.SPACE[Xini[NA:],Yini[NA:],Zini[NA:]] = 2    # type freeB
        
        # initial SPACE_REAL
        temp = np.empty((L,L,L))
        temp[:,:,:] = np.nan
        mc.SPACE_REAL = temp.tolist()
        for xyz in mc.A + mc.B:
            mc.SPACE_REAL[xyz[0]][xyz[1]][xyz[2]] = xyz.copy()

    # Initialize the system by source file
    else:
        # file format example: All_of5000step_in_nA1000_nB500_L40_U5.0_V10.0_chi2.0_isL.zip
        nstep  = int(inifile[ inifile.index('All_of')+6:inifile.index('step') ] ) 
        mc.NA  = int(inifile[ inifile.index('nA')+2:inifile.index('_nB') ] ) 
        mc.NB  = int(inifile[ inifile.index('nB')+2:inifile.index('_L') ] )
        mc.L   = int(inifile[ inifile.index('L')+1:inifile.index('_U') ] )
        mc.Ub1 = float(inifile[ inifile.index('U')+1:inifile.index('_V') ] )
        mc.Ub2 = float(inifile[ inifile.index('V')+1:inifile.index('_chi') ] )
        mc.chi = float(inifile[ inifile.index('chi')+3:inifile.index('_is') ]  )

        a2b_confg = inifile[ inifile.index('_is')+1:inifile.index('.zip')]
        if a2b_confg == 'isL':
            mc.A2B_isL = True
            mc.A2B_rod = False
        elif a2b_confg == 'isRod':
            mc.A2B_isL = False
            mc.A2B_rod = True
        elif a2b_confg == 'isBoth':        
            mc.A2B_isL = True
            mc.A2B_rod = True
        else:
            raise ValueError('A2B shape description is problematic')     

        # initialize A, B, AB, A2B lists  
        mc.A, mc.B, mc.AB, mc.A2B = io_control.read_a_snapshot_byfile(inifile)        

        # initialize SPACE and SPACE_REAL
        mc.SPACE = np.zeros((mc.L,mc.L,mc.L)) 
        for a in mc.A:
            mc.SPACE[ a[0] % mc.L ,a[1] % mc.L, a[2] % mc.L ] = 1
            mc.SPACE_REAL[ a[0] % mc.L][ a[1] % mc.L][ a[2] % mc.L ] = a
           
        for b in mc.B:
            mc.SPACE[ b[0] % mc.L,b[1] % mc.L, b[2] % mc.L ] = 2
            mc.SPACE_REAL[ b[0] % mc.L ][ b[1] % mc.L ][ b[2] % mc.L ] = b
        
        for ab in mc.AB:
            mc.SPACE[ ab[0][0] % mc.L,ab[0][1] % mc.L, ab[0][2] % mc.L ] = 3
            mc.SPACE[ ab[1][0] % mc.L,ab[1][1] % mc.L, ab[1][2] % mc.L ] = 4
            mc.SPACE_REAL[ ab[0][0] % mc.L][ab[0][1] % mc.L][ ab[0][2] % mc.L ] = ab
            mc.SPACE_REAL[ ab[1][0] % mc.L][ab[1][1] % mc.L][ ab[1][2] % mc.L ] = ab

        for a2b in mc.A2B:
            mc.SPACE[ a2b[0][0] % mc.L, a2b[0][1] % mc.L, a2b[0][2] % mc.L ] = 5
            mc.SPACE[ a2b[1][0] % mc.L, a2b[1][1] % mc.L, a2b[1][2] % mc.L ] = 6
            mc.SPACE[ a2b[2][0] % mc.L, a2b[2][1] % mc.L, a2b[2][2] % mc.L ] = 5
            mc.SPACE_REAL[ a2b[0][0] % mc.L][ a2b[0][1] % mc.L][ a2b[0][2] % mc.L ] = a2b
            mc.SPACE_REAL[ a2b[1][0] % mc.L][ a2b[1][1] % mc.L][ a2b[1][2] % mc.L ] = a2b
            mc.SPACE_REAL[ a2b[2][0] % mc.L][ a2b[2][1] % mc.L][ a2b[2][2] % mc.L ] = a2b
   




