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
   if "8_states" not in filename:
      continue

   if 'yorktown' in filename:
      device_name = 'yorktown'
   elif 'vigo' in filename:
      device_name = 'vigo'
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

   if device_name == 'no device':
      layout = 'None'
      circ = 'None'
   else:
      if 'layout-4-2-3' in filename:
         layout = '{4,2,3}'
         circ = 'True'
      elif 'layout-4-2-1' in filename:
         layout = '{4,2,1}'
         circ = 'False'
      elif 'layout_None' in filename:
         layout = 'None'
         circ = 'None'
      elif 'layout-0-1-2' in filename:
         layout = '{0,1,2}'
         circ = 'False'
      else:
         continue
         #raise ValueError
         #layout = 'None'
         #circ = 'None'

   n_shots = 10000
   n_states = 8

   base_dict = {'device' : device_name,
               'layout' : layout,
               'enc_type' : enc_type,
               'n_states' : n_states,
               'sim_type' : sim_type,
               'shots' : n_shots,
               'optimizer' : optimizer,
               'meas_mit' : meas_mit,
               'circ' : circ
               }

   print(base_dict)

   data = np.load(f"./{filename}")

   for energy in data:
      next_dict = base_dict
      next_dict['energy'] = energy
      df = df.append(next_dict, ignore_index=True)

print(df.groupby(['device','layout','enc_type','sim_type','n_states','shots','optimizer','meas_mit']).describe())

#colours = {"True" : "tab:blue", "False" : "tab:orange", "None" : "tab:gray"}
colours = {'vigo' : 'tab:blue', 'yorktown' : 'tab:orange', 'no device' : 'tab:gray'}

linestyles = {('True','vigo') : (0,(1,1)), ('False','vigo') : (0,(5,1)), ('None','no device') : '-.', ('True','yorktown') : (0,(1,5)), ('False','yorktown') : (0,(5,5))}

fig, ax = plt.subplots(figsize=(8,5))

for key, grp in df.groupby(['circ','meas_mit','enc_type','layout','device']):
    if key[2] == 'Jordan-Wigner':
        continue
    if key[1] == 'False':
       continue
    if key[0] == 'True':
        label = f'Loop: {key[3]}'
    elif key[0] == 'False':
        label = f'Line: {key[3]}'
    else:
        label = 'No noise'
    if key[4] == 'vigo':
        label = label + ' (V)'
    elif key[4] == 'yorktown':
        label = label + ' (Y)'
    sns.kdeplot(grp['energy'],bw='scott',label=f"{label}",color=colours[key[4]],linestyle=linestyles[(key[0],key[4])],ax=ax)

ax.axvline(x=diagonalized_values[8][1], color='black', label='True value (N = 8)', alpha=0.8)

handles, labels = ax.get_legend_handles_labels()
order = [0,1,3,2,4]
handles, labels = [handles[i] for i in order], [labels[i] for i in order]
ax.legend(handles,labels,fontsize=14)

ax.set_xlabel("Energy", fontsize=16)
#ax.set_xlim(-3,10)
#plt.ylim(0,20)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
ax.tick_params(labelsize=16)

#title_string = f"Yorktown, meas_mit={key[1]}"

#plt.title(title_string, fontsize=20)

fig.tight_layout()

plt.savefig(f"./8_states_yorktown.pdf")
plt.show()
