import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from colors import Colors
color = Colors.light_colors()  # Setting color style

def barplot(save_path=None,title=None, *arg,**kwarg):
    """just wrap seaborn barplot in nature style"""
    with plt.style.context(['science',"nature",'no-latex']):  # ,"ieee"
        

        fig, ax = plt.subplots()

        sns.barplot(palette=color,*arg,**kwarg)
        ax.tick_params(right=False,  top=False) 
        ax.tick_params(right=False, which='minor', top=False)
        ax.legend(title=None)
        ax.set(title = title)
        
        sns.despine()
        if save_path:
            plt.savefig(f'{save_path}.jpg',dpi=600)
            plt.savefig(f'{save_path}.svg',dpi=600)
    return ax

def distplot(save_path = None, *arg,**kwarg):
    """just wrap seaborn distplot in science style"""
    with plt.style.context(['science','no-latex']):  # ,"ieee"

        
        fig, ax = plt.subplots()
        sns.distplot(*arg,**kwarg)
        
        ax.tick_params(right=False,  top=False) 
        ax.tick_params(right=False, which='minor', top=False) 
        sns.despine()
        
        if save_path:
           fig.savefig(f'{save_path}.jpg',dpi=600)
           plt.savefig(f'{save_path}.svg',dpi=600)
    

def countplot(save_path=None, *arg,**kwarg):
    """just wrap seaborn countplot in nature style"""
    with plt.style.context(['science',"nature",'no-latex']):  # ,"ieee"
        

        fig, ax = plt.subplots()

        sns.countplot(palette=color,*arg,**kwarg)
        ax.tick_params(right=False,  top=False) 
        ax.tick_params(right=False, which='minor', top=False)

        sns.despine()
        if save_path:
            plt.savefig(f'{save_path}.jpg',dpi=600)
            plt.savefig(f'{save_path}.svg',dpi=600)