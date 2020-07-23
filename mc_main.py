# Jul 22, 2020 by caseypie
# - Can either start single or multiple systems with different initial NA, NB

# Mar 16, 2019 by caseypie
# - Monte Carlo lattice model algorithm for SynGAP-PSD95 phase separation system
# - Four types of molecules: A, B, AB, A2B

# Run MC simulation

import sys
import numpy as np
import mc_steps as mc
import initialize as ini
import parameter_setting as parset
import io_control
import pandas as pd
import multiprocessing as mp


# command: python3 mc_main.py [use pars file or not (Y/N)]  
#                             [ snapshot file (optional, if Y) ] 
#                             [NA] [NB] (optional, if N and only one system is simulated) 


# multiple system testing settings
nAmin, nAmax = 1, 20
nBmin, nBmax = 1, 2
scan_unit = 100
NAs = ((np.arange(nAmin-1,nAmax)+1)*scan_unit).astype(int)
NBs = ((np.arange(nBmin-1,nBmax)+1)*scan_unit).astype(int)    
NANBs = np.array( [ [x, y] for x in NAs for y in NBs ]) 


# MC period settings 
n_MCsteps = int(1e6)+1 # total number of MC steps
snap_shot_interval = int(1e3) # number of MC steps between two snapshots 


# Initialization
if sys.argv[1] == 'Y':
    try:
        ini.initialize_the_box( inifile=sys.argv[2] )
    except:
        ini.initialize_the_box()
     
    # Run MC
    info = (mc.NA, mc.NB, mc.L, mc.Ub1, mc.Ub2, mc.chi, mc.A2B_isL, mc.A2B_rod )
    MC_results_count = pd.DataFrame()

    print('Continue a single system simulation from file:')
    print(sys.argv[2])

    MCori = pd.DataFrame( \
                {'A nothing':[0], 'A move':[0], 'A+B->AB':[0], 'A+AB->A2B':[0], \
                 'B nothing': [0], 'B move':[0], 'B+A->AB':[0], \
                 'AB->A+B': [0], 'AB nothing':[0], 'AB move':[0], 'AB rotate':[0], \
                 'AB+A->A2B':[0], \
                 'A2B->A+AB':[0], 'A2B->AB+A':[0], 'A2B move':[0], 'A2B rotate':[0], \
                 'A2B cluster blocked':[0], 'A2B cluster':[0], 'A2B nothing':[0]  }  )
    MCcount = MCori.copy()
    rr = 'ini' 
    par_info, dirname = None, None 

    for i in range(n_MCsteps):  
        if i % int(snap_shot_interval) == 0:
            print('The ' + str(i) + 'th step: ', rr)    
            par_info, dirname = io_control.save_a_snapshot( \
                                 (mc.A, mc.B, mc.AB, mc.A2B), (i,) + info)
            MC_results_count = MC_results_count.append(MCcount)
            MCcount = MCori.copy()
 
        rr = mc.MC_step_select()
        MCcount[rr] += 1                 
  
    # output statistics of MC acceptance rate
    MC_results_count.to_csv( dirname + '/MCstat' + par_info + '.csv') 

elif sys.argv[1] == 'N': # import parameters here; may only use for scanning multiple (NA, NB) pairs:
    
    def ps_parallel(nanb):
        np.random.seed()
        pars = ( nanb[0], nanb[1], parset.L, \
                     parset.Ub1, parset.Ub2, parset.chi, \
                     parset.isL, parset.rod   )
     
        ini.initialize_the_box(pars=pars)
        
        MC_results_count = pd.DataFrame()
        MCori = pd.DataFrame( \
                   {'A nothing':[0], 'A move':[0], 'A+B->AB':[0], 'A+AB->A2B':[0], \
                    'B nothing': [0], 'B move':[0], 'B+A->AB':[0], \
                    'AB->A+B': [0], 'AB nothing':[0], 'AB move':[0], 'AB rotate':[0], \
                    'AB+A->A2B':[0], \
                    'A2B->A+AB':[0], 'A2B->AB+A':[0], 'A2B move':[0], 'A2B rotate':[0], \
                    'A2B cluster blocked':[0], 'A2B cluster':[0], 'A2B nothing':[0]  }  )
        MCcount = MCori.copy()
        par_info, dirname = None, None

        for i in range(n_MCsteps):  
            if i % int(snap_shot_interval) == 0:
                #print('The ' + str(i) + 'th step: ', rr)    
                par_info, dirname = io_control.save_a_snapshot( \
                                      (mc.A, mc.B, mc.AB, mc.A2B), (i,) + pars)
                MC_results_count = MC_results_count.append(MCcount)
                MCcount = MCori.copy()
 
            rr = mc.MC_step_select()
            MCcount[rr] += 1                 

        # output statistics of MC acceptance rate
        par_info = '_nA' + str(mc.NA) + '_nB' + str(mc.NB) + '_L' + str(mc.L) + \
                   '_U' + str(mc.Ub1) + '_V' + str(mc.Ub2) + '_chi' + str(mc.chi)

        if mc.A2B_isL == True and mc.A2B_rod == False:
            a2bconfg = 'isL'
        elif mc.A2B_isL == False and mc.A2B__rod == True:      
            a2bconfg = 'isRod'
        elif mc.A2B_isL == True and mc.A2B_rod == True:
            a2bconfg = 'isBoth' 
        else:
            raise ValueError('A2B shape description is problematic') 

        MC_results_count.to_csv( dirname + '/MCstat' + par_info + '.csv') 
    
    # single systme with assigned NA, NB
    if len(sys.argv) > 2:
        print('Start new single system simulation:')
        print('NA = ' + sys.argv[2] + ' , NB = ' + sys.argv[3]  )
        print('L = ' + str(parset.L) + ' , U = ' + str(parset.Ub1) + ' , V = ' + str(parset.Ub2) + ' , chi = ' + str(parset.chi) )
        print('A2B_isL: ' , parset.isL )
        print('A2B_rod: ' , parset.rod )
        NANB = [ int(sys.argv[2]), int(sys.argv[3])]
        ps_parallel(NANB) 
    # multiple system testing, parallel by mp pool
    else:
        print('Start new multiple system simulation:')
        print('NA: from ' + str( int(nAmin*scan_unit) ) + ' to ' + str( int(nAmax*scan_unit)  )   )
        print('NB: from ' + str( int(nBmin*scan_unit) ) + ' to ' + str( int(nBmax*scan_unit)  )   )
        print('L = ' + str(parset.L) + ' , U = ' + str(parset.Ub1) + ' , V = ' + str(parset.Ub2) + ' , chi = ' + str(parset.chi) )
        print('A2B_isL: ' , parset.isL )
        print('A2B_rod: ' , parset.rod )
        pool = mp.Pool(processes=40)
        pool.map(ps_parallel, NANBs) 

else:
    raise ValueError('Please indicate whether to use parameter file or not.')
