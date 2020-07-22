# Mar 16, 2019 by caseypie
# - Monte Carlo lattice model algorithm for SynGAP-PSD95 phase separation system
# - Four types of molecules: A, B, AB, A2B

# Binding and move/rotation algorithms:
# A:   move->(if succeeded)->check binding A+B->AB or A+AB->A2B
# B:   move->(if succeeded)->check binding B+A->AB
# AB:  decoupling AB->A+B -> (if failed) -> move/rotation -> (if succeeded) -> binding AB+A->A2B
# A2B: decoupling A2B->AB+A -> (if failed) -> move/rotation -> (if failed) -> cluster move

# Variable naming rules:
# 1. Variables with '_REA' are real-space coordinates in numpy array 
# 2. Variables with '_PBC' are PBC coordinates in numpy array
# 3. Variables without subscription label are real-space coordinates in Python list
# 4. Generally, PBC coordinates are all in numpy array; Python list is only used for
#    real-space coordinates
# 5. Use '_temp' for exceptions of rule no. 4.

import numpy as np

# Variables that will be defined by initialize module

L = None    # MC cubic box edge length
Ub1 = None  # binding energy of A+B->AB
Ub2 = None  # binding energy of AB+A->A2B
chi = None  # phase separation Flory-Huggins parameter between each pair of A-B in two A2B

A2B_isL = None   # True: A-B-A must be in L-shape conformation
A2B_rod = None  # True: A-B-A must be in linear-shape conformation 

SPACE = None      # The PBC space storing occupation states of lattice sites
SPACE_REAL = None # The PBC space sotring real-space coordinates of molecules 

NA  = None  # number of free A
NB  = None  # number of free B

A   = None  # list of all free A 
B   = None  # list of all free B
AB  = None  # list of all AB
A2B = None  # list of all A2B




#--------------------------------------------------------------------------------------------

# Main Monte Carlo function
def MC_step_select():
    nA, nB, nAB, nA2B = len(A), len(B), len(AB), len(A2B)
    r_pick =  np.random.randint(0,nA+nB+nAB+nA2B)
    if r_pick < nA:
        action = MC_step_freeA(r_pick)
    elif r_pick < nA+nB:
        action = MC_step_freeB(r_pick-nA)
    elif r_pick < nA+nB+nAB:
        action = MC_step_AB(r_pick-nA-nB)
    else:
        action = MC_step_A2B(r_pick-nA-nB-nAB)
        
    return action
        
#--------------------------------------------------------------------------------------------
    
def MC_step_freeA(iA):
    # locate
    A_ori_REA = np.array(A[iA])
    A_ori_PBC = Real2PBC(A_ori_REA)
    
    #print(iA, A_ori_REA, A_ori_PBC)
    
    # move
    r_mov = np.random.randint(0,6)
    A_mov_REA = threeDmove(A_ori_REA, r_mov)
    A_mov_PBC = Real2PBC(A_mov_REA)
    
    #print('mov:', r_mov,A_mov_REA ,A_mov_PBC )
    
    # no particle on the to-move site: move it
    if SPACE[A_mov_PBC[0],A_mov_PBC[1],A_mov_PBC[2]] == 0:
        # update SPACE
        SPACE[A_ori_PBC[0],A_ori_PBC[1],A_ori_PBC[2]] = 0
        SPACE[A_mov_PBC[0],A_mov_PBC[1],A_mov_PBC[2]] = 1
        # update A list
        A[iA] = A_mov_REA.tolist()
        # update SPACE_REAL
        SPACE_REAL[A_ori_PBC[0]][A_ori_PBC[1]][A_ori_PBC[2]] = np.nan
        SPACE_REAL[A_mov_PBC[0]][A_mov_PBC[1]][A_mov_PBC[2]] = A[iA].copy()      
    # movement failed: MC step ends
    else:
        return 'A nothing'
      
    # binding after successful movement
    nn_PBC = Real2PBC( np.array( [Xmove1(A[iA])] + [Xmove2(A[iA])] + \
                                 [Ymove1(A[iA])] + [Ymove2(A[iA])] + \
                                 [Zmove1(A[iA])] + [Zmove2(A[iA])]     ) )
    nn_ind = np.where( np.logical_not((SPACE[nn_PBC[:,0],nn_PBC[:,1],nn_PBC[:,2]]!=2) \
                                     *(SPACE[nn_PBC[:,0],nn_PBC[:,1],nn_PBC[:,2]]!=4))  )[0]
    nn_CanBind_PBC = nn_PBC[nn_ind,:]
    nn_num = nn_CanBind_PBC.shape[0]

    # if there is an A or AB in n.n.: may bind it
    if nn_num > 0:
        if nn_num == 1:
            nn_tg_PBC = nn_CanBind_PBC[0] # binding target 
        else:
            # pick up one of the n.n. bindable particle
            r_bind = np.random.randint(nn_num)
            nn_tg_PBC = nn_CanBind_PBC[r_bind]   # binding target   
            
        # A+B->AB
        if SPACE[nn_tg_PBC[0],nn_tg_PBC[1],nn_tg_PBC[2]]==2:
            # update SPACE
            SPACE[A_mov_PBC[0],A_mov_PBC[1],A_mov_PBC[2]]=3
            SPACE[nn_tg_PBC[0],nn_tg_PBC[1],nn_tg_PBC[2]]=4        
            # update A, B and AB lists 
            Bnn = SPACE_REAL[nn_tg_PBC[0]][nn_tg_PBC[1]][nn_tg_PBC[2]].copy()
            ABnew = [A[iA].copy()] + [Bnn.copy()]
            AB.append(ABnew.copy())             
            del A[iA]
            B.remove(Bnn)
            # update SPACE_REAL
            SPACE_REAL[A_mov_PBC[0]][A_mov_PBC[1]][A_mov_PBC[2]] = ABnew.copy()
            SPACE_REAL[nn_tg_PBC[0]][nn_tg_PBC[1]][nn_tg_PBC[2]] = ABnew.copy()
            return 'A+B->AB'
        # A+AB->A2B
        else: #if SPACE[nn[0],nn[1],nn[2]]==4:
            # check the geometry of the molecule
            ABnn = SPACE_REAL[nn_tg_PBC[0]][nn_tg_PBC[1]][nn_tg_PBC[2]].copy()
            ABnn_PBC = Real2PBC(np.array(ABnn))
            ABaxis   = np.nonzero( np.array( \
                        [ ABnn_PBC[0,0]-ABnn_PBC[1,0], \
                          ABnn_PBC[0,1]-ABnn_PBC[1,1], \
                          ABnn_PBC[0,2]-ABnn_PBC[1,2] ] ) )[0]
                
            A2Baxis  = np.nonzero( np.array( \
                        [ A_mov_PBC[0]-ABnn_PBC[1,0], \
                          A_mov_PBC[1]-ABnn_PBC[1,1], \
                          A_mov_PBC[2]-ABnn_PBC[1,2]  ] ) )[0]
            
            if ABaxis.shape[0] != 1 or A2Baxis.shape[0] != 1:
                print(SPACE[nn_tg_PBC[0],nn_tg_PBC[1],nn_tg_PBC[2]], \
                      ABnn_PBC, A_mov_PBC, \
                      ABaxis, A2Baxis      )
                raise ValueError('Error: AB molecules are broken!(A)')
            
            # do not allow rod-shape but this A-B-A is rod-shape: no binding 
            if not(A2B_rod) and A2Baxis[0] == ABaxis[0]:
                return 'A move'#'A: A-B-A rod-shape geometry'
    
            # do not allow L-shape but this A-B-A is L-shape: no binding 
            if not(A2B_isL) and A2Baxis[0] != ABaxis[0]:
                return 'A move'#'A: A-B-A L-shape geometry'

            # geometrty test passed: binding
            # update SPACE
            SPACE[A_mov_PBC[0],A_mov_PBC[1],A_mov_PBC[2]]=5
            SPACE[ABnn_PBC[0,0],ABnn_PBC[0,1],ABnn_PBC[0,2]]=5     
            SPACE[ABnn_PBC[1,0],ABnn_PBC[1,1],ABnn_PBC[1,2]]=6 
            # update A, AB and A2B lists           
            A2Bnew = ABnn.copy() + [A[iA].copy()]
            A2B.append(A2Bnew.copy())   
            del A[iA]
            AB.remove(ABnn)
            # update SPACE_REAL
            SPACE_REAL[A_mov_PBC[0]][A_mov_PBC[1]][A_mov_PBC[2]] = A2Bnew.copy()
            SPACE_REAL[ABnn_PBC[0,0]][ABnn_PBC[0,1]][ABnn_PBC[0,2]] = A2Bnew.copy()
            SPACE_REAL[ABnn_PBC[1,0]][ABnn_PBC[1,1]][ABnn_PBC[1,2]] = A2Bnew.copy()
            return 'A+AB->A2B'
    # if no A or AB in n.n.: no binding
    else:
        return 'A move'
    
#--------------------------------------------------------------------------------------------

def MC_step_freeB(iB):
    # locate
    B_ori_REA = np.array(B[iB])
    B_ori_PBC = Real2PBC(B_ori_REA)
    
    # move
    r_mov = np.random.randint(0,6)
    B_mov_REA = threeDmove(B_ori_REA, r_mov)
    B_mov_PBC = Real2PBC(B_mov_REA)
    
    # no particle on the to-move site: move it
    if SPACE[B_mov_PBC[0],B_mov_PBC[1],B_mov_PBC[2]] == 0:
        # update SPACE
        SPACE[B_ori_PBC[0],B_ori_PBC[1],B_ori_PBC[2]] = 0
        SPACE[B_mov_PBC[0],B_mov_PBC[1],B_mov_PBC[2]] = 2
        # update A list
        B[iB] = B_mov_REA.tolist()
        # update SPACE_REAL
        SPACE_REAL[B_ori_PBC[0]][B_ori_PBC[1]][B_ori_PBC[2]] = np.nan
        SPACE_REAL[B_mov_PBC[0]][B_mov_PBC[1]][B_mov_PBC[2]] = B[iB].copy()    
    # movement failed: MC step ends
    else:
        return 'B nothing'
         
    # binding
    nn_PBC = Real2PBC( np.array( [Xmove1(B[iB])] + [Xmove2(B[iB])] + \
                                 [Ymove1(B[iB])] + [Ymove2(B[iB])] + \
                                 [Zmove1(B[iB])] + [Zmove2(B[iB])]     ) )
    nn_ind = np.where(SPACE[nn_PBC[:,0],nn_PBC[:,1],nn_PBC[:,2]]==1)[0]
    nn_CanBind_PBC = nn_PBC[nn_ind,:]
    nn_num = nn_CanBind_PBC.shape[0]
    
    # if there is an A in n.n.: bind it
    if nn_num > 0:
        if nn_num == 1:
            nn_tg_PBC = nn_CanBind_PBC[0] # binding target
        else:
            # pick up one of the n.n. bindable particle
            r_bind = np.random.randint(nn_num)
            nn_tg_PBC = nn_CanBind_PBC[r_bind]  # binding target   
            
        # B+A->AB
        # update SPACE
        SPACE[B_mov_PBC[0],B_mov_PBC[1],B_mov_PBC[2]]=4
        SPACE[nn_tg_PBC[0],nn_tg_PBC[1],nn_tg_PBC[2]]=3
        # update A, B and AB lists
        #print(nn_test)
        Ann = SPACE_REAL[nn_tg_PBC[0]][nn_tg_PBC[1]][nn_tg_PBC[2]].copy()        
        ABnew = [Ann.copy()] + [B[iB].copy()]
        AB.append(ABnew.copy())    
        del B[iB]
        A.remove(Ann)
        # update SPACE_REAL
        SPACE_REAL[nn_tg_PBC[0]][nn_tg_PBC[1]][nn_tg_PBC[2]] = ABnew.copy()
        SPACE_REAL[B_mov_PBC[0]][B_mov_PBC[1]][B_mov_PBC[2]] = ABnew.copy()
        return 'B+A->AB'
    # if no A in n.n.: no binding
    else:
        return 'B move'
        
#--------------------------------------------------------------------------------------------

def MC_step_AB(iAB):
    # locate
    AB_ori_REA = np.array(AB[iAB])
    AB_ori_PBC = Real2PBC(AB_ori_REA)
  
    # MC test for unbinding
    r_unbound = np.random.rand()
    if r_unbound < np.exp(-Ub1):
        # AB -> A + B
        # update SPACE
        SPACE[AB_ori_PBC[0,0],AB_ori_PBC[0,1],AB_ori_PBC[0,2]]=1
        SPACE[AB_ori_PBC[1,0],AB_ori_PBC[1,1],AB_ori_PBC[1,2]]=2
        # update A, B and AB lists 
        Anew, Bnew = AB_ori_REA[0,:].tolist(), AB_ori_REA[1,:].tolist() 
        A.append( Anew.copy() )
        B.append( Bnew.copy() )
        del AB[iAB]
        # update SPACE_REAL
        SPACE_REAL[AB_ori_PBC[0,0]][AB_ori_PBC[0,1]][AB_ori_PBC[0,2]] = Anew.copy()
        SPACE_REAL[AB_ori_PBC[1,0]][AB_ori_PBC[1,1]][AB_ori_PBC[1,2]] = Bnew.copy()
        return 'AB->A+B'# This MC step ends
            
    # if unbound failed: move or rotate
    r_move_rotate = np.random.randint(2) # 0 for rotate, 1 for move
    # move
    if r_move_rotate:
        r_mov = np.random.randint(0,6)
        A_mov_REA = threeDmove(AB_ori_REA[0,:], r_mov)
        A_mov_PBC = Real2PBC(A_mov_REA)
        B_mov_REA = threeDmove(AB_ori_REA[1,:], r_mov)
        B_mov_PBC = Real2PBC(B_mov_REA)

        # check to_move site(s)
        if np.array_equal(A_mov_PBC, AB_ori_PBC[1,:]): # if A_mov == B_ori
            # only have to check whether B_mov is empty or not
            to_mov_PBC = np.expand_dims(B_mov_PBC, axis=0)  
        elif np.array_equal(B_mov_PBC,AB_ori_PBC[0,:]): # if B_mov == A_ori
            # only have to check whether A_mov is empty or not
            to_mov_PBC = np.expand_dims(A_mov_PBC, axis=0) 
        else: # have to check both A_mov and B_mov
            to_mov_PBC = np.concatenate((A_mov_PBC, B_mov_PBC)).reshape((2,3)) 
        
        #print(to_mov_PBC,'AB')
        # no particle on all of the to-move sites: move it
        if np.count_nonzero(SPACE[to_mov_PBC[:,0],to_mov_PBC[:,1],to_mov_PBC[:,2]]) == 0: 
            # update SPACE
            SPACE[AB_ori_PBC[:,0],AB_ori_PBC[:,1],AB_ori_PBC[:,2]] = 0
            SPACE[A_mov_PBC[0],A_mov_PBC[1],A_mov_PBC[2]] = 3
            SPACE[B_mov_PBC[0],B_mov_PBC[1],B_mov_PBC[2]] = 4
            # update AB list
            AB[iAB] = [A_mov_REA.tolist()] + [B_mov_REA.tolist()]
            # update SPACE_REAL
            for xyz in AB_ori_PBC:
                SPACE_REAL[xyz[0]][xyz[1]][xyz[2]] = np.nan
            SPACE_REAL[A_mov_PBC[0]][A_mov_PBC[1]][A_mov_PBC[2]] = AB[iAB].copy()
            SPACE_REAL[B_mov_PBC[0]][B_mov_PBC[1]][B_mov_PBC[2]] = AB[iAB].copy()
        # movement failed: MC step ends
        else:
            return 'AB nothing'
        
    # rotate
    else:
        # use B as rotate origin
        r_rotate = np.random.randint(0,6)
        A_rot_REA = threeDrotate(AB_ori_REA[0,:]-AB_ori_REA[1,:], r_rotate) + AB_ori_REA[1,:]
        A_rot_PBC = Real2PBC(A_rot_REA)
        # no particle on the to-rotate site: rotate it        
        if SPACE[A_rot_PBC[0],A_rot_PBC[1],A_rot_PBC[2]] == 0:
            # update SPACE
            SPACE[AB_ori_PBC[0,0],AB_ori_PBC[0,1],AB_ori_PBC[0,2]] = 0
            SPACE[A_rot_PBC[0],A_rot_PBC[1],A_rot_PBC[2]] = 3
            # update AB list
            AB[iAB][0] = A_rot_REA.tolist()
            # update SPACE_REAL: only A changes location   
            SPACE_REAL[AB_ori_PBC[0,0]][AB_ori_PBC[0,1]][AB_ori_PBC[0,2]] = np.nan
            SPACE_REAL[AB_ori_PBC[1,0]][AB_ori_PBC[1,1]][AB_ori_PBC[1,2]] = AB[iAB].copy()
            SPACE_REAL[A_rot_PBC[0]][A_rot_PBC[1]][A_rot_PBC[2]] = AB[iAB].copy()     
        # rotation failed: MC step ends
        else:
            return 'AB nothing'
         
    # AB+A->A2B binding after successful movement/rotation
    nn_PBC = Real2PBC( np.array( [Xmove1(AB[iAB][1])] + [Xmove2(AB[iAB][1])] + \
                                 [Ymove1(AB[iAB][1])] + [Ymove2(AB[iAB][1])] + \
                                 [Zmove1(AB[iAB][1])] + [Zmove2(AB[iAB][1])]     ) )
    nn_ind = np.where(SPACE[nn_PBC[:,0],nn_PBC[:,1],nn_PBC[:,2]]==1)[0]
    nn_CanBind_PBC = nn_PBC[nn_ind,:] 
    nn_num = nn_CanBind_PBC.shape[0]
    
    #if there is free A around the B in AB after movement or rotation: bind it 
    if nn_num > 0:
        if nn_num == 1:
            nn_tg_PBC = nn_CanBind_PBC[0] # binding target
        else:
            # pick up one of the n.n. bindable particle
            r_bind = np.random.randint(nn_num)
            nn_tg_PBC = nn_CanBind_PBC[r_bind] # binding target
    
        # check the geometry of the molecule
        AB_PBC  = Real2PBC(np.array(AB[iAB]))        
        ABaxis  = np.nonzero( np.array( \
                    [ AB_PBC[0,0]-AB_PBC[1,0], \
                      AB_PBC[0,1]-AB_PBC[1,1], \
                      AB_PBC[0,2]-AB_PBC[1,2] ] ) )[0]
        A2Baxis = np.nonzero( np.array( \
                    [ nn_tg_PBC[0]-AB_PBC[1,0], \
                      nn_tg_PBC[1]-AB_PBC[1,1], \
                      nn_tg_PBC[2]-AB_PBC[1,2] ] ) )[0]

        if ABaxis.shape[0] != 1 or A2Baxis.shape[0] != 1:
            print(ABaxis, A2Baxis)
            raise ValueError('Error: AB molecules are broken!(AB)')

        # do not allow rod-shape but this A-B-A is rod-shape: no binding 
        if not(A2B_rod) and A2Baxis[0] == ABaxis[0]:
            #return 'AB: A-B-A rod-shape geomerty'
            if r_move_rotate: 
                return 'AB move'
            else:
                return 'AB rotate'  
 
        # do not allow L-shape but this A-B-A is L-shape: no binding 
        if not(A2B_isL) and A2Baxis[0] != ABaxis[0]:
            #return 'AB: A-B-A L-shape geomerty'
            if r_move_rotate: 
                return 'AB move'
            else:
                return 'AB rotate'  
  
        #print(ABaxis, A2Baxis,  AB_PBC, nn_tg_PBC)
    
        # geometrty test passed: binding 
        # update SPACE
        SPACE[AB_PBC[0,0],AB_PBC[0,1],AB_PBC[0,2]]=5
        SPACE[AB_PBC[1,0],AB_PBC[1,1],AB_PBC[1,2]]=6
        SPACE[nn_tg_PBC[0],nn_tg_PBC[1],nn_tg_PBC[2]]=5         
        # update A, AB and A2B lists 
        Ann = SPACE_REAL[nn_tg_PBC[0]][nn_tg_PBC[1]][nn_tg_PBC[2]].copy()
        A2Bnew = AB[iAB].copy() + [Ann.copy()]
        A2B.append(A2Bnew.copy())    
        del AB[iAB]
        A.remove(Ann)
        # update SPACE_REAL
        SPACE_REAL[AB_PBC[0,0]][AB_PBC[0,1]][AB_PBC[0,2]] = A2Bnew.copy()
        SPACE_REAL[AB_PBC[1,0]][AB_PBC[1,1]][AB_PBC[1,2]] = A2Bnew.copy()        
        SPACE_REAL[nn_tg_PBC[0]][nn_tg_PBC[1]][nn_tg_PBC[2]] = A2Bnew.copy()     
        return 'AB+A->A2B'
    # if no A in the n.n. of the B in AB: no binding
    else:
        if r_move_rotate: 
            return 'AB move'
        else:
            return 'AB rotate'  
 
        
#--------------------------------------------------------------------------------------------   
    
def MC_step_A2B(iA2B):    
    # locate
    A2B_ori_REA = np.array(A2B[iA2B])
    A2B_ori_PBC = Real2PBC(A2B_ori_REA)
  
    # MC test for unbinding
    r_unbound = np.random.rand()
    if r_unbound < np.exp(-Ub2):
        # A2B -> AB + A
        r_Aunb = np.random.randint(2) # 0 for unbinding 1st A and 1 for 2nd A
        if r_Aunb == 0:
            # update SPACE
            SPACE[A2B_ori_PBC[0,0],A2B_ori_PBC[0,1],A2B_ori_PBC[0,2]]=1 # free A
            SPACE[A2B_ori_PBC[1,0],A2B_ori_PBC[1,1],A2B_ori_PBC[1,2]]=4 # B in AB 
            SPACE[A2B_ori_PBC[2,0],A2B_ori_PBC[2,1],A2B_ori_PBC[2,2]]=3 # A in AB
            # update A, AB, and A2B lists
            Anew, ABnew = A2B_ori_REA[0,:].tolist(), A2B_ori_REA[:0:-1,:].tolist() 
            A.append( Anew.copy() )
            AB.append( ABnew.copy() )
            del A2B[iA2B]
            # update SPACE_REAL
            SPACE_REAL[A2B_ori_PBC[0,0]][A2B_ori_PBC[0,1]][A2B_ori_PBC[0,2]] = Anew.copy()
            SPACE_REAL[A2B_ori_PBC[1,0]][A2B_ori_PBC[1,1]][A2B_ori_PBC[1,2]] = ABnew.copy()
            SPACE_REAL[A2B_ori_PBC[2,0]][A2B_ori_PBC[2,1]][A2B_ori_PBC[2,2]] = ABnew.copy()
            return 'A2B->A+AB' # This MC step ends         
        else:
            # update SPACE
            SPACE[A2B_ori_PBC[0,0],A2B_ori_PBC[0,1],A2B_ori_PBC[0,2]]=3 # A in AB
            SPACE[A2B_ori_PBC[1,0],A2B_ori_PBC[1,1],A2B_ori_PBC[1,2]]=4 # B in AB 
            SPACE[A2B_ori_PBC[2,0],A2B_ori_PBC[2,1],A2B_ori_PBC[2,2]]=1 # free A
            # update A, AB, and A2B lists
            Anew, ABnew = A2B_ori_REA[2,:].tolist(), A2B_ori_REA[0:2,:].tolist()  
            A.append( Anew.copy() )
            AB.append( ABnew.copy() )
            del A2B[iA2B]
            # update SPACE_REAL
            SPACE_REAL[A2B_ori_PBC[0,0]][A2B_ori_PBC[0,1]][A2B_ori_PBC[0,2]] = ABnew.copy()
            SPACE_REAL[A2B_ori_PBC[1,0]][A2B_ori_PBC[1,1]][A2B_ori_PBC[1,2]] = ABnew.copy()   
            SPACE_REAL[A2B_ori_PBC[2,0]][A2B_ori_PBC[2,1]][A2B_ori_PBC[2,2]] = Anew.copy()
            return 'A2B->AB+A' # This MC step ends   
    
    # if unbound failed: single-A2B move or rotate
      
    # chi energy before move/rotate
    Uchi_ori = -chi*A2B_touch_num(A2B[iA2B], SPACE)
                         
    # coordinates after movement or rotation
    r_move_rotate = np.random.randint(2) # 0 for rotate, 1 for move
    # move
    if r_move_rotate:        
        r_move = np.random.randint(0,6)
        A1_mov_REA = threeDmove(A2B_ori_REA[0,:], r_move)
        A1_mov_PBC = Real2PBC(A1_mov_REA)
        B_mov_REA = threeDmove(A2B_ori_REA[1,:], r_move)
        B_mov_PBC = Real2PBC(B_mov_REA)
        A2_mov_REA = threeDmove(A2B_ori_REA[2,:], r_move)
        A2_mov_PBC = Real2PBC(A2_mov_REA)
        
        # check the to-move site(s) 
        if np.array_equal(A1_mov_PBC , A2B_ori_PBC[1,:]) or \
           np.array_equal( B_mov_PBC , A2B_ori_PBC[2,:]): # A1_mov = B_ori or B_mov = A2_ori
            if not(np.array_equal(A1_mov_PBC , A2B_ori_PBC[1,:] ) ): # only B_mov = A2_ori
                to_mov_PBC = np.concatenate((A1_mov_PBC,A2_mov_PBC)).reshape((2,3))
            elif not(np.array_equal( B_mov_PBC , A2B_ori_PBC[2,:]) ): # only A1_mov = B_ori
                to_mov_PBC = np.concatenate((B_mov_PBC,A2_mov_PBC)).reshape((2,3))            
            else: # A1_mov = B_ori and B_mov = A2_ori      
                to_mov_PBC = np.expand_dims( A2_mov_PBC, axis=0) 
        elif np.array_equal( B_mov_PBC , A2B_ori_PBC[0,:] ) or \
           np.array_equal(A2_mov_PBC , A2B_ori_PBC[1]): # B_mov = A1_ori or A2_mov = B_ori
            if not( np.array_equal(B_mov_PBC, A2B_ori_PBC[0,:])): # only A2_mov = B_ori
                to_mov_PBC = np.concatenate((A1_mov_PBC,B_mov_PBC)).reshape((2,3))
            elif not(np.array_equal(A2_mov_PBC ,A2B_ori_PBC[1,:])): # only B_mov = A1_ori
                to_mov_PBC = np.concatenate((A1_mov_PBC,A2_mov_PBC)).reshape((2,3))
            else:
                to_mov_PBC = np.expand_dims( A1_mov_PBC, axis=0 )
        else: # all A,A,B are moving
            to_mov_PBC = np.concatenate((A1_mov_PBC,B_mov_PBC,A2_mov_PBC)).reshape((3,3))
            
        # no particle on the to-move site(s): may move
        if np.count_nonzero(SPACE[to_mov_PBC[:,0],to_mov_PBC[:,1],to_mov_PBC[:,2]]) == 0:  
            # copy a SPACE for move testing
            NewSPACE = SPACE.copy()
            # update NewSPACE
            NewSPACE[A2B_ori_PBC[:,0],A2B_ori_PBC[:,1],A2B_ori_PBC[:,2]] = 0
            NewSPACE[A1_mov_PBC[0],A1_mov_PBC[1],A1_mov_PBC[2]] = 5
            NewSPACE[B_mov_PBC[0], B_mov_PBC[1], B_mov_PBC[2] ] = 6
            NewSPACE[A2_mov_PBC[0],A2_mov_PBC[1],A2_mov_PBC[2]] = 5
         
            # chi energy after move
            A2Bnew_temp = [A1_mov_PBC.tolist()]+[B_mov_PBC.tolist()]+[A2_mov_PBC.tolist()]
            Uchi_after = -chi*A2B_touch_num(A2Bnew_temp, NewSPACE)
            del NewSPACE
            # MC test for energy
            r_u = np.random.rand() 
            # test passed: move it 
            # note: always passed when Uchi_ori > Uchi_after
            if r_u < np.exp(Uchi_ori-Uchi_after): 
                # update SPACE
                SPACE[A2B_ori_PBC[:,0],A2B_ori_PBC[:,1],A2B_ori_PBC[:,2]] = 0
                SPACE[A1_mov_PBC[0],A1_mov_PBC[1],A1_mov_PBC[2]] = 5
                SPACE[B_mov_PBC[0] ,B_mov_PBC[1] ,B_mov_PBC[2] ] = 6
                SPACE[A2_mov_PBC[0],A2_mov_PBC[1],A2_mov_PBC[2]] = 5
                # update A2B list
                A2B[iA2B] = [A1_mov_REA.tolist()]+[B_mov_REA.tolist()]+[A2_mov_REA.tolist()]
                # update SPACE_REAL
                for xyz in A2B_ori_PBC:
                    SPACE_REAL[xyz[0]][xyz[1]][xyz[2]] = np.nan
                SPACE_REAL[A1_mov_PBC[0]][A1_mov_PBC[1]][A1_mov_PBC[2]] = A2B[iA2B].copy()
                SPACE_REAL[B_mov_PBC[0] ][B_mov_PBC[1] ][B_mov_PBC[2] ] = A2B[iA2B].copy()
                SPACE_REAL[A2_mov_PBC[0]][A2_mov_PBC[1]][A2_mov_PBC[2]] = A2B[iA2B].copy()            
                return 'A2B move'          
    # rotate
    else:
        # use B as rotate origin
        r_rotate = np.random.randint(0,6)
        A1_rot_REA = threeDrotate(A2B_ori_REA[0,:]-A2B_ori_REA[1,:], r_rotate) + A2B_ori_REA[1,:]
        A1_rot_PBC = Real2PBC(A1_rot_REA)
        A2_rot_REA = threeDrotate(A2B_ori_REA[2,:]-A2B_ori_REA[1,:], r_rotate) + A2B_ori_REA[1,:]
        A2_rot_PBC = Real2PBC(A2_rot_REA)
        
        # no particle on the to-rotate sites: may rotate        
        if SPACE[A1_rot_PBC[0],A1_rot_PBC[1],A1_rot_PBC[2]] == 0 and \
           SPACE[A2_rot_PBC[0],A2_rot_PBC[1],A2_rot_PBC[2]] == 0:
            # copy a SPACE for move testing
            NewSPACE = SPACE.copy()
            # update NewSPACE
            NewSPACE[A2B_ori_PBC[0,0],A2B_ori_PBC[0,1],A2B_ori_PBC[0,2]] = 0
            NewSPACE[A2B_ori_PBC[2,0],A2B_ori_PBC[2,1],A2B_ori_PBC[2,2]] = 0
            NewSPACE[A1_rot_PBC[0],A1_rot_PBC[1],A1_rot_PBC[2]] = 5
            NewSPACE[A2_rot_PBC[0],A2_rot_PBC[1],A2_rot_PBC[2]] = 5                
            # chi energy after rotate
            A2Bnew_temp = [A1_rot_PBC.tolist()]+[A2B_ori_PBC[1,:].tolist()]+[A2_rot_PBC.tolist()]
            Uchi_after = -chi*A2B_touch_num(A2Bnew_temp, NewSPACE)
            del NewSPACE
            # MC test for energy
            r_u = np.random.rand() 
            # test passed: rotate it 
            # note: always passed when Uchi_ori > Uchi_after
            if r_u < np.exp(Uchi_ori-Uchi_after): 
                # update SPACE
                SPACE[A2B_ori_PBC[0,0],A2B_ori_PBC[0,1],A2B_ori_PBC[0,2]] = 0
                SPACE[A2B_ori_PBC[2,0],A2B_ori_PBC[2,1],A2B_ori_PBC[2,2]] = 0
                SPACE[A1_rot_PBC[0],A1_rot_PBC[1],A1_rot_PBC[2]] = 5
                SPACE[A2_rot_PBC[0],A2_rot_PBC[1],A2_rot_PBC[2]] = 5  
                # update A2B list
                A2B[iA2B] = [A1_rot_REA.tolist()] + \
                            [A2B_ori_REA[1,:].tolist()] + \
                            [A2_rot_REA.tolist()]
                # update SPACE_REAL
                for xyz in A2B_ori_PBC:
                    SPACE_REAL[xyz[0]][xyz[1]][xyz[2]] = np.nan
                SPACE_REAL[A1_rot_PBC[0]][A1_rot_PBC[1]][A1_rot_PBC[2]] = A2B[iA2B].copy()
                SPACE_REAL[A2_rot_PBC[0]][A2_rot_PBC[1]][A2_rot_PBC[2]] = A2B[iA2B].copy()
                SPACE_REAL[A2B_ori_PBC[1,0]][A2B_ori_PBC[1,1]][A2B_ori_PBC[1,2]] \
                                                                        = A2B[iA2B].copy()
                return 'A2B rotate'      

    # If single-A2B move/rotate failed: cluster movement
    NewSPACE = SPACE.copy()
    clus_ori = Cluster_Search(A2B[iA2B].copy(), NewSPACE )
    del NewSPACE
    nA2Bclus = len(clus_ori)
    if nA2Bclus == 1: # only one A2B: this is not a cluster
        return 'A2B cluster blocked' 
    
    # A block for testing Cluster Search
    """
    test = np.reshape(clus_ori, (3*nA2Bclus,3) )
    test_PBC = np.concatenate( ( Xmove1Numpy(test), Xmove2Numpy(test) , \
                                 Ymove1Numpy(test), Ymove2Numpy(test) , \
                                 Zmove1Numpy(test), Zmove2Numpy(test) ,  ) ) % L
    for tt in test_PBC:
        xxx = SPACE_REAL[tt[0]][tt[1]][tt[2]]
        if np.array(xxx).shape == (3,3):
            if not(xxx in clus_ori):
                print(xxx, clus_ori)
                raise ValueError('cluster wrong')
    """    
    
    # movement direction and coordinates 
    r_mov =np.random.randint(0,6)
    clus_ori_REA = np.reshape( clus_ori, ( 3*nA2Bclus,3) )
    clus_ori_PBC = Real2PBC(clus_ori_REA)
    clus_mov_REA = threeDmoveNumpy( clus_ori_REA, r_mov )
    clus_mov_PBC = Real2PBC(clus_mov_REA)
        
    # label the to-move A2B
    SPACE[clus_ori_PBC[:,0],clus_ori_PBC[:,1],clus_ori_PBC[:,2]] = 7
    SPACE[clus_ori_PBC[1::3,0],clus_ori_PBC[1::3,1],clus_ori_PBC[1::3,2]] = 8 

    # check whether all to-move points are empty
    n0 = np.where(SPACE[clus_mov_PBC[:,0],clus_mov_PBC[:,1],clus_mov_PBC[:,2]]==0)[0].shape[0]
    n7 = np.where(SPACE[clus_mov_PBC[:,0],clus_mov_PBC[:,1],clus_mov_PBC[:,2]]==7)[0].shape[0]
    n8 = np.where(SPACE[clus_mov_PBC[:,0],clus_mov_PBC[:,1],clus_mov_PBC[:,2]]==8)[0].shape[0]
        
    if n0+n7+n8 == nA2Bclus*3: # all to-move points are empty: move it
        # update SPACE
        SPACE[clus_ori_PBC[:,0],clus_ori_PBC[:,1],clus_ori_PBC[:,2]] = 0
        SPACE[clus_mov_PBC[:,0],clus_mov_PBC[:,1],clus_mov_PBC[:,2]] = 5
        SPACE[clus_mov_PBC[1::3,0],clus_mov_PBC[1::3,1],clus_mov_PBC[1::3,2]] = 6            
        # update A2B list
        clus_mov = np.reshape(clus_mov_REA,(nA2Bclus, 3, 3)).tolist()
        #print(clus_mov)
        for ixyz, A2Bxyz in enumerate(clus_mov):
            ith = A2B.index( clus_ori[ixyz] )
            A2B[ith] = A2Bxyz.copy()
        # update SPACE_REAL
        for xxyyzz in clus_ori_PBC:
            SPACE_REAL[xxyyzz[0]][xxyyzz[1]][xxyyzz[2]] = np.nan
        for A2Bxyz in clus_mov:
            SPACE_REAL[A2Bxyz[0][0] % L ][A2Bxyz[0][1] % L ][A2Bxyz[0][2] % L ] = A2Bxyz.copy()
            SPACE_REAL[A2Bxyz[1][0] % L ][A2Bxyz[1][1] % L ][A2Bxyz[1][2] % L ] = A2Bxyz.copy()
            SPACE_REAL[A2Bxyz[2][0] % L ][A2Bxyz[2][1] % L ][A2Bxyz[2][2] % L ] = A2Bxyz.copy()    
        return 'A2B cluster'
    else: # not all to-move sites are empty: no cluster movement
        SPACE[clus_ori_PBC[:,0],clus_ori_PBC[:,1],clus_ori_PBC[:,2]] = 5
        SPACE[clus_ori_PBC[1::3,0],clus_ori_PBC[1::3,1],clus_ori_PBC[1::3,2]] = 6
        return 'A2B nothing' 
        #print('A2B cluster blocked')
    
        
        
        
#--------------------------------------------------------------------------------------------

# convert a coordinate in real space to be in periodic boundary condition (PBC) 
# xyz is a numpy array!
def Real2PBC(xyz):
    #xyz_p = [ xyz[i] % L for i in range(3) ]
    #return xyz_p
    return xyz % L

# output target coordinate of a 3D translational movement 
# xyz is a numpy array
def threeDmoveNumpy(xyz, direction):
    switcher = {0: Xmove1Numpy,
                1: Xmove2Numpy,
                2: Ymove1Numpy,
                3: Ymove2Numpy,
                4: Zmove1Numpy,
                5: Zmove2Numpy}
    func = switcher.get(direction, lambda xyz: 'nothing')
    return func(xyz)
        

def Xmove1Numpy(xyz):
    b = xyz.copy()
    b[:,0] += 1
    return b

def Xmove2Numpy(xyz):
    b = xyz.copy()
    b[:,0] -= 1
    return b

def Ymove1Numpy(xyz):
    b = xyz.copy()
    b[:,1] += 1
    return b

def Ymove2Numpy(xyz):
    b = xyz.copy()
    b[:,1] -= 1
    return b

def Zmove1Numpy(xyz):
    b = xyz.copy()
    b[:,2] += 1
    return b

def Zmove2Numpy(xyz):
    b = xyz.copy()
    b[:,2] -= 1
    return b



def threeDmove(xyz, direction):
    switcher = {0: Xmove1,
                1: Xmove2,
                2: Ymove1,
                3: Ymove2,
                4: Zmove1,
                5: Zmove2}
    func = switcher.get(direction, lambda xyz: 'nothing')
    return func(xyz)
        

def Xmove1(xyz):
    b = xyz.copy()
    b[0] += 1
    return b

def Xmove2(xyz):
    b = xyz.copy()
    b[0] -= 1
    return b

def Ymove1(xyz):
    b = xyz.copy()
    b[1] += 1
    return b

def Ymove2(xyz):
    b = xyz.copy()
    b[1] -= 1
    return b

def Zmove1(xyz):
    b = xyz.copy()
    b[2] += 1
    return b

def Zmove2(xyz):
    b = xyz.copy()
    b[2] -= 1
    return b



# output target coordinate of a 3D 90-degree rotation 
def threeDrotate(xyz, axis):
    switcher = {0: Xrotate1,
                1: Xrotate2,
                2: Yrotate1,
                3: Yrotate2,
                4: Zrotate1,
                5: Zrotate2}
    func = switcher.get(axis, lambda xyz: 'nothing')
    return func(xyz)

Xrotp90 = np.array([[1,0,0],[0,0,-1],[0,1,0]])
Xrotm90 = np.array([[1,0,0],[0,0,1],[0,-1,0]])
Yrotp90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])
Yrotm90 = np.array([[0,0,-1],[0,1,0],[1,0,0]])
Zrotp90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
Zrotm90 = np.array([[0,1,0],[-1,0,0],[0,0,1]])

# xyz is a numpy array
def Xrotate1(xyz):
    return np.dot( Xrotp90, xyz )
def Xrotate2(xyz):
    return np.dot( Xrotm90, xyz  )
def Yrotate1(xyz):
    return np.dot( Yrotp90, xyz  )
def Yrotate2(xyz):
    return np.dot( Yrotm90, xyz  )
def Zrotate1(xyz):
    return np.dot( Zrotp90, xyz  )
def Zrotate2(xyz):
    return np.dot( Zrotm90, xyz  )
    
# xyzA2B is a list
def A2B_touch_num(xyzA2B, theSPACE):
    chi_touch = 0
    A1, B, A2 = xyzA2B[0], xyzA2B[1], xyzA2B[2]
    A1nn_PBC = Real2PBC( np.array( [Xmove1(A1)] + [Xmove2(A1)] + \
                               [Ymove1(A1)] + [Ymove2(A1)] + \
                               [Zmove1(A1)] + [Zmove2(A1)] )   )
    chi_touch += np.where(theSPACE[A1nn_PBC[:,0],\
                                   A1nn_PBC[:,1],\
                                   A1nn_PBC[:,2]] == 6 )[0].shape[0] # n.n. of A1 is a B in A2B
    
    Bnn_PBC =  Real2PBC( np.array( [Xmove1(B)] + [Xmove2(B)] + \
                               [Ymove1(B)] + [Ymove2(B)] + \
                               [Zmove1(B)] + [Zmove2(B)] )   )
    chi_touch += np.where(theSPACE[Bnn_PBC[:,0],\
                                   Bnn_PBC[:,1],\
                                   Bnn_PBC[:,2]] == 5 )[0].shape[0] # n.n. of B is an A in A2B 
                            
    A2nn_PBC = Real2PBC( np.array( [Xmove1(A2)] + [Xmove2(A2)] + \
                               [Ymove1(A2)] + [Ymove2(A2)] + \
                               [Zmove1(A2)] + [Zmove2(A2)] )   )
    chi_touch += np.where(theSPACE[A2nn_PBC[:,0],\
                                   A2nn_PBC[:,1],\
                                   A2nn_PBC[:,2]] == 6 )[0].shape[0] # n.n. of A2 is a B in A2B
                            
    return chi_touch #- 4 # subtract the two intra-molecular interactions

# Cluster search
# xyzA2B is a list
def Cluster_Search(xyzA2B, NewSPACE):
    A2Bincluster = [xyzA2B.copy()]
    A2B_PBC = np.array(xyzA2B) % L 
    NewSPACE[A2B_PBC[:,0], A2B_PBC[:,1],A2B_PBC[:,2]] = 0
    One_Cluster_Search(xyzA2B, A2Bincluster, NewSPACE)
    return A2Bincluster
    
    
# xyzA2B is a list 
def One_Cluster_Search(xyzA2B, A2Bincluster, NewSPACE):    
    # search n.n. of xyzA2B
    A2B_REA = np.array(xyzA2B)
    A2Bnn_PBC = np.concatenate( ( Xmove1Numpy(A2B_REA), Xmove2Numpy(A2B_REA) , \
                                  Ymove1Numpy(A2B_REA), Ymove2Numpy(A2B_REA) , \
                                  Zmove1Numpy(A2B_REA), Zmove2Numpy(A2B_REA) ,  ) ) % L
    for xyz in A2Bnn_PBC:
        if NewSPACE[xyz[0],xyz[1],xyz[2]] == 5 or NewSPACE[xyz[0],xyz[1],xyz[2]] == 6:
            A2Bnext = SPACE_REAL[xyz[0]][xyz[1]][xyz[2]].copy()
            A2Bincluster.append( A2Bnext.copy() )
            A2Bnext_PBC = np.array(A2Bnext) % L
            NewSPACE[A2Bnext_PBC[:,0],A2Bnext_PBC[:,1],A2Bnext_PBC[:,2]] = 0
            One_Cluster_Search(A2Bnext.copy(), A2Bincluster, NewSPACE)
