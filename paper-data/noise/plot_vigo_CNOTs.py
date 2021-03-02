#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
HOME=os.path.expanduser("~")
sys.path.append(HOME)
from mylib_python import myplt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

plt.rcParams = myplt.set_style(unit="in")
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(9,6)})
sns.set(font_scale=1.6)
#sns.set()
colors = myplt.colors()
nums = [0,1,2,3]
rs = ["ibmq_vigo"]
ms=10
with sns.axes_style("whitegrid"):
    fig, axs = myplt.set_canvas(r=1,c=1,width=9,height=6,shy=True,shx=True,unit="in")
    ins = inset_axes(axs, width="50%", height="35%", bbox_to_anchor=(-0.25, 0.2, 0.8, 0.7), bbox_transform=axs.transAxes)
    for r in rs:
        x = []
        y = []
        y_e = []
        c = "orange"
        for num in nums:
            filename = "./outputs-qasm-device-CNOTs-jw/2020-07-29_jordan_wigner-4_states-qasm_simulator-10000_shots-SPSA-"+str(r)+"_layout-2-1-3-4-mit_meas-CNOTs"+str(num)+"-energies.npy"
            #filename = "./outputs-qasm-device-CNOTs-jw/2020-08-11_jordan_wigner-4_states-qasm_simulator-10000_shots-SPSA-"+str(r)+"_layout-2-1-3-4-mit_meas-CNOTs"+str(num)+"-energies.npy"
            data = np.load(filename)
            e = data.mean()
            x.append(2*num+1)
            y.append(data.mean())
            y_e.append(data.std())
        axs.errorbar( x, y, y_e, color=c, marker="o", ms=ms, ls = "", label="Noise:"+str(r) )
        coef, cov = np.polyfit(x,y,1,cov=True)
        axs.errorbar( [0,], [coef[-1],], yerr=[np.sqrt(cov[-1,-1]),], markersize=ms, marker="o", color=c, mfc="none")
        poly1d_fn = np.poly1d(coef)
        axs.plot( [0,max(x)], poly1d_fn([0,max(x)]), color=c, ls = "-" )
        ins.errorbar( x, y, y_e, color=c, marker="o", ms=ms, ls = "", label="Noise:"+str(r) )
        ins.errorbar( [0,], [coef[-1],], yerr=[np.sqrt(cov[-1,-1]+sum( [x**2 for x in y_e] )/len(y_e)),], markersize=ms, marker="o", color=c, mfc="none")
        axs.errorbar( [0,], [coef[-1],], yerr=[np.sqrt(cov[-1,-1]+sum( [x**2 for x in y_e] )/len(y_e)),], markersize=ms, marker="o", color=c, mfc="none")
        print(coef[-1])
        ins.plot( [0,max(x)], poly1d_fn([0,max(x)]), color=c, ls = "-" )

    for r in rs:
        x = []
        y = []
        y_e = []
        c = "blue"
        for num in nums:
            filename = "outputs-qasm-device-CNOTs-gc/2020-07-29_gray_code-4_states-qasm_simulator-10000_shots-SPSA-"+str(r)+"_layout-2-1-mit_meas-CNOTs"+str(num)+"-energies.npy"
            #filename = "outputs-qasm-device-CNOTs-gc/2020-08-11_gray_code-4_states-qasm_simulator-10000_shots-SPSA-"+str(r)+"_layout-2-1-mit_meas-CNOTs"+str(num)+"-energies.npy"
            data = np.load(filename)
            e = data.mean()
            x.append(2*num+1)
            y.append(data.mean())
            y_e.append(data.std())
        axs.errorbar( x, y, y_e, color=c, marker="o", ms=ms, ls = "", label="Noise:"+str(r) )
        coef, cov = np.polyfit(x,y,1,cov=True)
        axs.errorbar( [0,], [coef[-1],], yerr=[np.sqrt(cov[-1,-1]),], marker="o", markersize=ms, color=c, mfc="none")
        poly1d_fn = np.poly1d(coef)
        axs.plot( [0,max(x)], poly1d_fn([0,max(x)]), color=c, ls = "-" )
        ins.errorbar( x, y, y_e, color=c, marker="o", ms=ms, ls = "", label="Noise:"+str(r) )
        #ins.errorbar( [0,], [coef[-1],], yerr=[np.sqrt(cov[-1,-1]),], markersize=ms, marker="o", color=c, mfc="none")
        ins.errorbar( [0,], [coef[-1],], yerr=[np.sqrt(cov[-1,-1]+sum( [x**2 for x in y_e] )/len(y_e)),], markersize=ms, marker="o", color=c, mfc="none")
        axs.errorbar( [0,], [coef[-1],], yerr=[np.sqrt(cov[-1,-1]+sum( [x**2 for x in y_e] )/len(y_e)),], markersize=ms, marker="o", color=c, mfc="none")
        print(coef[-1])
        ins.plot( [0,max(x)], poly1d_fn([0,max(x)]), color=c, ls = "-" )

    #axs.legend()
    axs.axhline(y=-2.143, ls="-", c="k")
    ins.axhline(y=-2.143, ls="-", c="k")
    #axs.annotate(r"Jordan-Wigner encoding", color="orange", xy = (4,0.6), rotation="30")
    axs.annotate(r"One-hot encoding", color="orange", xy = (4,0.6), rotation="30")
    axs.annotate("Exact", xy = (6,-2.1), c="k")
    axs.set_ylabel("Energy (MeV)")
    axs.axhline(y=-2.143, ls="-", c="k")
    axs.annotate(r"Gray code encoding", color="blue", xy = (2,-1.7), rotation="6")
    axs.set_xlabel("2 (Number of CNOT pairs) + 1")
    axs.set_xticks(range(-1,10))
    ins.set_xticks(range(-1,10))
    axs.set_yticks(np.arange(-3,5,1))
    ins.set_yticks(np.arange(-3,2,0.2))
    axs.set_yticks(np.arange(-3,2,0.25),minor=True)
    ins.set_yticks(np.arange(-3,2,0.05),minor=True)
    axs.set_xlim(-0.5,8)
    axs.set_ylim(-2.5,4)
    ins.set_xlim(-0.25,2)
    ins.set_ylim(-2.2,-1.6)
    #axs.indicate_inset_zoom(ins)
plt.tight_layout()
plt.savefig("extrapolation_zero_noise.pdf")
