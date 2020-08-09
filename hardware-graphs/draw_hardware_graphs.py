import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

import networkx as nx

scale = { 'ns' : 1.5e6, 'nc' : 15, 'ew' : 2e3, 'ec' : 30} 

vigo = { 'name' : 'ibmq_vigo',
         'nodes' : range(0,5),
         'node_pos' : {0:(0,1), 1:(1,2), 2:(1,1), 3:(2,1), 4:(1,0)},
         'edges' : [(0,1), (1,2), (1,3), (3,4)],
         'node_label_pos' : {0:(0.34,-0.05), 1:(0.34,0.05), 2:(0.02,-0.2), 3:(-0.34,-0.02), 4:(0.36,-0.05)}
      }
yorktown = { 'name' : 'ibmq_5_yorktown - ibmqx2',
             'nodes' : range(0,5),
             'node_pos' : {0:(0,1), 1:(1,2), 2:(1,1), 3:(2,1), 4:(1,0)},
             'edges' : [(0,1), (1,2), (0,2), (2,3), (3,4), (2,4)],
         'node_label_pos' : {0:(0.34,-0.18), 1:(0.34,0.05), 2:(0.33,-0.15), 3:(-0.31,0.14), 4:(0.34,-0.05)}
      }

devices = [vigo, yorktown]

for device in devices:

   # read calibrations file
   calibrations = pd.read_csv(f"calibrations/{device['name']}.csv")
   print(calibrations.info())
   meas_err = calibrations.iloc[:,4].values
   single_q_err = calibrations.iloc[:,5].values
   print('meas_err:',meas_err)
   print('single_q_err:',single_q_err)
   two_q_err_raw = calibrations.iloc[:,6]
   
   # hideous method for getting two-qubit error values for edges
   two_q_err = []
   for i,e in two_q_err_raw.items():
      print(e)
      esp1 = e.split(sep=', ')
      es = ""
      for j, je in enumerate(esp1):
         esp2 = je.split(sep=': ')
         es = es + f"\"{esp2[0]}\": {esp2[1]}, "
      es = es.strip(', ')
      print(f"{{{es}}}")
      two_q_err.append(json.loads(f"{{{es}}}"))

   print('two_q_err:',two_q_err)

   # Scale node sizes proportional to single-q error rate
   print('err1:', single_q_err)
   node_sizes = single_q_err * scale['ns']
   print('sizes:', node_sizes)

   # Scale node colour inversely proportional to meas err
   node_color = (1-meas_err*scale['nc'])
   print('node_color:', node_color)

   # Scale edge width proportional and colour inversely proportional to two-qubit error
   err2list = [two_q_err[edge[0]][f"cx{edge[0]}_{edge[1]}"] for edge in device['edges']]
   err2_arr = np.asarray(err2list)
   print('err2:', err2_arr)
   widths = err2_arr * scale['ew']
   print('widths:', widths)
   edge_color = (1-err2_arr*scale['ec'])
   print('edge_color:', edge_color)

   edge_cmap = plt.cm.Greys
   edge_labels = {}
   for i,edge in enumerate(device['edges']):
      edge_labels[edge] = f"{err2list[i]*100:4.2}%"

   node_strs = {n: str(n) for n in device['nodes']} 

   # Set range of values and create colourmap/norm object
   cm = plt.cm.Blues # Change colourmap here
   norm = mpl.colors.Normalize(vmin=0, vmax=1)

   graph = nx.Graph()
   pos = device['node_pos']
   node_list = list(device['nodes'])
   edge_list = device['edges']

   # shift position of measurement and single-q error relative to center of nodes   
   print(pos)
   #print({key,val for key,val in pos})
   node_err_pos = {}
   nlp = device['node_label_pos']
   for key,val in pos.items():
         node_err_pos[key] = (val[0]+nlp[key][0],val[1]+nlp[key][1])
   err_labels = {}
   for n in node_list:
      err_labels[n] = f"({single_q_err[n]*100:4.2}%, {meas_err[n]*100:4.2}%)"
   plt.figure()
   #plt.figure(figsize=(10,10))

   # Draw the graph
   nx.draw(graph, pos, node_color=node_color, node_size=node_sizes, width=widths, nodelist=node_list, edgelist=edge_list, with_labels=True, linewidths=10, cmap=cm, vmin=0, vmax=1, labels=node_strs, edge_color=edge_color, edge_cmap=edge_cmap, edge_vmin=0, edge_vmax=1)
   nx.draw_networkx_labels(graph, pos=node_err_pos, labels=err_labels)
   nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

   # Get colourbars on plot
   #sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
   #plt.colorbar(sm)
   #sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
   #plt.colorbar(sm)

   # Add extra margins so all labels are visible
   #extra_margin = 0.05
   #right_margin = 0.53
   #x0, x1, y0, y1 = plt.axis()
   #plt.axis((x0-extra_margin,
   #          x1+extra_margin+right_margin,
   #          y0-extra_margin,
   #          y1+extra_margin
   #        ))
   #plt.tight_layout(rect=[0,0,1.2,1])

   plt.savefig(f"{device['name']}_graph.pdf")
   plt.show()
