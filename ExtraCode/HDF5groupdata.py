import numpy as np 
import h5py 

cass_jumping = np.loadct(open("C:/Users/Cassie G/Documents/390FinalProject-1/jumping_cg.csv"), delimiter=",", skiprows=1)
cass_walking = np.loadct(open("C:/Users/Cassie G/Documents/390FinalProject-1/walking_cg.csv"), delimiter=",", skiprows=1)

leo_jumping = np.loadct(open("C:/Users/Cassie G/Documents/390FinalProject-1/jumping_lp.csv"), delimiter=",", skiprows=1)
leo_walking = np.loadct(open("C:/Users/Cassie G/Documents/390FinalProject-1/walking_lp.csv"), delimiter=",", skiprows=1)

henry_jumping = np.loadct(open("C:/Users/Cassie G/Documents/390FinalProject-1/jumping_hc.csv"), delimiter=",", skiprows=1)
henry_walking = np.loadct(open("C:/Users/Cassie G/Documents/390FinalProject-1/walking_hc.csv"), delimiter=",", skiprows=1)

#writing to hdf5
with h5py.File('./hdf5_data.h5' , 'w') as hdf:
    #defining the individual groups on the LHS for Cassie Greidanus
    #Jump and Walk are stored separate under Cassie Greidanus folder
    CG_dataj = hdf.create_group('CassieGreidanus/Jump')
    CG_dataw = hdf.create_group('CassieGreidanus/Walk')

    CG_dataj.create_dataset('cassJump', data=cass_jumping)
    CG_dataw.create_dataset('cassWalk', data=cass_walking)

    #defining the individual groups on the LHS for Leo
    #Jump and Walk are stored separate under Leo's folder
    LP_dataj = hdf.create_group('LeoPalerma/Jump')
    LP_dataw = hdf.create_group('LeoPalerma/Walk')

    LP_dataj.create_dataset('leoJump', data=leo_jumping)
    LP_dataw.create_dataset('leoWalk', data=leo_walking)

    #defining the individual groups on the LHS for Henry
    #Jump and Walk are stored separate under Henry's folder
    HC_dataj = hdf.create_group('HenryChen/Jump')
    HC_dataw = hdf.create_group('HenryChen/Walk')

    HC_dataj.create_dataset('henryJump', data=henry_jumping)
    HC_dataw.create_dataset('henryWalk', data=henry_walking)


#reading to hdf5
'''
with h5py.File('.hdf5_data.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print(ls)
    cassJump = hdf.get('cassJump')
    print(dpe(cassJump))
    cJump_array = np.array(cassJump)
    print(dpe(cJump_array))
'''