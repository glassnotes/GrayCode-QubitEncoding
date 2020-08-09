#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import sys
import os

import pandas as pd
from scipy import stats

HEADERS = ['device', 'layout', 'enc_type', 'n_states', 'sim_type', 'shots', 'optimizer', 'energy', 'meas_mit']

df = pd.DataFrame(columns = HEADERS)


diagonalized_values = [
        (0, 0),
        (1,  -0.43658111),
        (2, -1.749160),
        (3, -2.045671), 
        (4, -2.1439810), 
        (5, -2.183592), 
        (6, -2.201568), 
        (7, -2.210416), 
        (8, -2.215038),
        (16, -2.221059)
] 

for filename in os.listdir("."):
   if "bkp" in filename:
      continue
   if "energies.npy" not in filename:
      continue
   if "4_states" not in filename:
      continue

   if 'vigo' in filename:
      device_name = 'vigo'
   elif 'yorktown' in filename:
      device_name = 'yorktown'
   elif 'no_device' in filename:
      device_name = 'no device'
   else:
      continue
   print(filename)

   enc_type = 'Gray code' if 'gray_code' in filename else 'Jordan-Wigner'
   optimizer = 'SPSA' if 'SPSA' in filename else 'Nelder-Mead'
   sim_type = 'QASM' if 'qasm' in filename else statevector
   meas_mit = 'True' if 'mit_meas' in filename else 'False'
   if device_name == 'no device':
      meas_mit = 'None'

   if 'layout-2-1-3-4' in filename:
      layout = '2-1-3-4'
   if 'layout-0-1-2-3' in filename:
      layout = '0-1-2-3'
   elif 'layout-2-1' in filename:
      layout = '2-1'
   elif 'layout_None' in filename:
      layout = 'None'
   else:
      #continue
      #raise ValueError
      layout = 'None'

   n_shots = 10000
   n_states = 4

   base_dict = {'device' : device_name,
               'layout' : layout,
               'enc_type' : enc_type,
               'n_states' : n_states,
               'sim_type' : sim_type,
               'shots' : n_shots,
               'optimizer' : optimizer,
               'meas_mit' : meas_mit
               }

   print(base_dict)

   data = np.load(f"{filename}")

   for energy in data:
      next_dict = base_dict
      next_dict['energy'] = energy
      df = df.append(next_dict, ignore_index=True)

print(df.groupby(['device','layout','enc_type','sim_type','n_states','shots','optimizer','meas_mit']).describe())

#colours = {'Gray code' : "tab:blue", 'Jordan-Wigner' : "tab:orange", 'None' : "tab:gray"}
colours = {'vigo' : 'tab:blue', 'yorktown' : 'tab:orange', 'no device' : 'tab:gray'}

linestyles = {('True','vigo') : (0,(5,1)), ('False','vigo') : (0,(1,1)), ('None','no device') : '-.', ('True','yorktown') : (0,(5,5)), ('False','yorktown') : (0,(1,5))}
 
for key, grp in df.groupby('enc_type'):
    fig, ax = plt.subplots(figsize=(8,5))

    print(key) 
    for mit_key, mit_grp in grp.groupby(['meas_mit','layout','device']):
         print(mit_key)
         print(mit_grp.describe())
         if mit_key[0] == 'False':
             label = 'Noise'
         elif mit_key[0] == 'True':
             label = 'Noise w/ mitigation'
         elif mit_key[0] == 'None':
             label = 'No noise'
         else:
             raise ValueError
         if mit_key[2] == 'vigo':
            label = label + ' (V)'
         elif mit_key[2] == 'yorktown':
            label = label + ' (Y)'
         #plt.hist(mit_grp['energy'])
         sns.kdeplot(mit_grp['energy'], bw='scott', label=label,color=colours[mit_key[2]],linestyle=linestyles[(mit_key[0],mit_key[2])], ax=ax)

    ax.axvline(x=diagonalized_values[4][1], color='black', label='True value (N = 4)', alpha=0.8)

    if key == 'Gray code':
        ax.legend(fontsize=16, loc='lower right')
    else:
        ax.legend_.remove()

    ax.set_xlabel("Energy", fontsize=16)
    ax.set_xlim(-3,1.5)
    plt.ylim(0,6)
    #ax.set_xticks(fontsize=16)
    #ax.set_yticks(fontsize=16)
    ax.tick_params(labelsize=16)

    title_string = f"Encoding: {key}"

    ax.set_title(title_string, fontsize=20)

    fig.tight_layout()

    plt.savefig(f"4_state: {key}.pdf")
    plt.show()
