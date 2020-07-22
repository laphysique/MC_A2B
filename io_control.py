# Mar 16, 2019 by caseypie
# - Monte Carlo lattice model algorithm for SynGAP-PSD95 phase separation system
# - Four types of molecules: A, B, AB, A2B

# Auxiliary functions for input/output simulation results

import numpy as np
import zipfile as zpf
import os
import multiprocessing as mp

# read a snapshot
def read_a_snapshot_byfile(fname):
    Aloc, Bloc, ABloc, A2Bloc, _ = read_a_snapshot(0,0,0,0,0,0,0, fname=fname)
    return Aloc, Bloc, ABloc, A2Bloc
    
# read a snapshot
# info = ( istep, nA, nB, Lsize, u1, u2, chi_r, isL, is_rod )
# dir_and_file = (dirname, fname)
# file format example: All_of5000step_in_nA1000_nB500_L40_U5.0_V10.0_chi2.0_isL.zip
def read_a_snapshot( info=None, dir_and_file=None ):
    try:
        istep, nA, nB, Lsize, u1, u2, chi_r, isL, is_rod = info
        if isL == True and is_rod == False:
            a2bconfg = 'isL'
        elif isL == False and is_rod == True:      
            a2bconfg = 'isRod'
        elif isL == True and is_rod == True:
            a2bconfg = 'isBoth' 
        else:
            raise ValueError('A2B shape description is problematic')  

        par_info = '_nA' + str(nA) + '_nB' + str(nB) + '_L' + str(Lsize) + \
                   '_U' + str(u1) + '_V' + str(u2) + '_chi' + str(chi_r) + \
                   '_' + a2bconfg
 
        dirname = './results/' + a2bconfg + '/All_steps_in' + par_info
        fname = 'All_of' + str(istep) + 'step_in' + par_info +  '.zip'
    except:
        dirname, fname = dir_and_file  
    
    zfiles = np.load(dirname + '/' + fname )
    ffname = fname[fname.index('_of'):fname.index('.zip')]    
          
    Ax = zfiles['A' + ffname + '.txt']    
    Aloc = np.fromstring(Ax[(Ax.index(b'\n')):].decode("utf-8"), sep=' ' , dtype=int)
    if np.array_equal(Aloc, np.zeros(1) ):
        Aloc = np.array([])    
    else:
        Aloc = Aloc.reshape( (int(Aloc.shape[0]/3),3) ) 

    Bx = zfiles['B' + ffname + '.txt']
    Bloc = np.fromstring(Bx[(Bx.index(b'\n')):].decode("utf-8"), sep=' ' , dtype=int)
    if np.array_equal(Bloc, np.zeros(1) ):
        Bloc = np.array([]) 
    else:
        Bloc = Bloc.reshape( (int(Bloc.shape[0]/3),3) )


    ABx = zfiles['AB' + ffname + '.txt']
    ABloc = np.fromstring(ABx[(ABx.index(b'\n')):].decode("utf-8"), sep=' ' , dtype=int)
    if np.array_equal(ABloc, np.zeros(1) ):
        ABloc = np.array([]) 
    else:
        ABloc = ABloc.reshape( (int(ABloc.shape[0]/6),6) ) 

    A2Bx = zfiles['A2B' + ffname + '.txt']
    A2Bloc = np.fromstring(A2Bx[(A2Bx.index(b'\n')):].decode("utf-8"), sep=' ' , dtype=int)
    if np.array_equal(A2Bloc, np.zeros(1) ):    
        A2Bloc = np.array([]) 
    else:
        A2Bloc = A2Bloc.reshape( (int(A2Bloc.shape[0]/9),9) ) 

   
    return Aloc, Bloc, ABloc, A2Bloc, 'All' + fname

# read-a-snapshot function for parallel import
def ras_par(dandf):
    return read_a_snapshot( dir_and_file=dandf)

# read all snapshots in a directory
def read_all_snapshots(dirname ):    
    # format of dirname: All_steps_in_nA1000_nB500_L40_U5.0_V10.0_chi2.0_isL
    NA  = int(dirname[ dirname.index('_nA')+3:dirname.index('_nB') ] ) 
    NB  = int(dirname[ dirname.index('_nB')+3:dirname.index('_L') ] )
    L   = int(dirname[ dirname.index('_L')+2:dirname.index('_U') ] )
    Ub1 = float(dirname[ dirname.index('_U')+2:dirname.index('_V') ] )
    Ub2 = float(dirname[ dirname.index('_V')+2:dirname.index('_chi') ] )
    chi = float(dirname[ dirname.index('_chi')+4:dirname.index('_is') ]  )
    is_L_rod = dirname[ dirname.index('_is')+1:-1]

    info = (NA,NB,L,Ub1,Ub2,chi, is_L_rod)

    print('filedir:',dirname)

    all_snapshots = os.listdir(dirname)
    ind, ft = [], []
  
    print('number of files:', len(all_snapshots))
    for i, fname in enumerate(all_snapshots):
        if '.zip' in fname:
            ind.append(i)
            ft.append( int(fname[fname.index('_of')+3:fname.index('step')]) )

    ind_read = np.argsort(ft)
 
    all_reads = [ ( dirname, all_snapshots[ind[i]]) for i in ind_read ]
 

    pool = mp.Pool(processes=4)
    Alocs, Blocs, ABlocs, A2Blocs, _ = zip(*pool.map( ras_par, all_reads ))
    
    """
    Alocs, Blocs, ABlocs, A2Blocs = [],[],[],[]
    for i in ind_read: 
        Aloc, Bloc, ABloc, A2Bloc, _ = \
              read_a_snapshot(dir_and_file=(dirname, all_snapshots[ind[i]]) )      
        Alocs.append(Aloc)
        Blocs.append(Bloc)
        ABlocs.append(ABloc)
        A2Blocs.append(A2Bloc)        
    """
    

    return Alocs, Blocs, ABlocs, A2Blocs, info, np.array(ft)[ind_read]



# write a snapshot to file
# All_Mol = (A, B, AB, A2B)
# info = ( istep, nA, nB, Lsize, u1, u2, chi_r, isL, is_rod )
def save_a_snapshot( All_Mol, info, dirname=None):
    Aloc, Bloc, ABloc, A2Bloc = All_Mol

    istep, nA, nB, Lsize, u1, u2, chi_r, isL, is_rod = info
    if isL == True and is_rod == False:
        a2bconfg = 'isL'
    elif isL == False and is_rod == True:      
        a2bconfg = 'isRod'
    elif isL == True and is_rod == True:
        a2bconfg = 'isBoth' 
    else:
        raise ValueError('A2B shape description is problematic') 
  
    par_info = '_nA' + str(nA) + '_nB' + str(nB) + '_L' + str(Lsize) + \
               '_U' + str(u1) + '_V' + str(u2) + '_chi' + str(chi_r) + \
               '_' + a2bconfg   

    if dirname==None:
        dirname = './results/' + a2bconfg + \
                  '/All_steps_in' + par_info

    if os.path.isdir(dirname):
        pass
    else:
        os.makedirs(dirname)

    ABi = np.reshape(ABloc, (len(ABloc), 3*2)  )
    A2Bi = np.reshape(A2Bloc, (len(A2Bloc), 3*3)  )
    fname = '_of' + str(istep) + 'step_in' + par_info
    fileA   = 'A' + fname + '.txt'
    fileB   = 'B' + fname + '.txt'
    fileAB  = 'AB' + fname + '.txt'
    fileA2B = 'A2B' + fname + '.txt'

    np.savetxt(fileA  , np.array(Aloc), fmt='%d', header='A: x-y-z' )
    np.savetxt(fileB  , np.array(Bloc), fmt='%d', header='B: x-y-z')
    np.savetxt(fileAB , ABi,  fmt='%d', header='A: x-y-z ; B: x-y-z')
    np.savetxt(fileA2B, A2Bi, fmt='%d', header='A1: x-y-z ; B: x-y-z ; A2: x-y-z' )
    with zpf.ZipFile(dirname + '/All' + fname + '.zip' , 'w') as myzip:
        myzip.write(fileA)
        myzip.write(fileB)
        myzip.write(fileAB)
        myzip.write(fileA2B)

    os.remove(fileA)
    os.remove(fileB)
    os.remove(fileAB)
    os.remove(fileA2B)
 
    return par_info, dirname
