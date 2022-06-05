"""Table of Content
    #TODO ieee 格式
    1. lines_plot 
    2. 
    overview the gallery https://github.com/garrettj403/SciencePlots/wiki/Gallery
"""
import matplotlib.pyplot as plt
import numpy as np



def lines_plot( X, Y=None ,xyparam= None,**pparam):
    """Plot lines togethear in scientific style, see more color circle in 
       https://github.com/garrettj403/SciencePlots/wiki/Gallery 

    Args:
        x (list): x-axis ID values
        Y (list[ list ]): y-axis values-list
        xyparam：
            xylables (dict(xlabels=,ylabel =), optional): _description_. Defaults to None.
            can add more properties , e.g. title, xlim, ylim, xscale, yscale...
        style (list, optional): style name list, can add more color circle. Defaults to ['science',"nature",'no-latex'].
        pparam：
            save_path (str, optional): basename of save path, without suffix. Defaults to None.
            legend_title (str, optional): _description_. Defaults to None.
            legend_labels (list, optional): _description_. Defaults to None.
    """    
    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    # todo FIX this bug, unanno these will cause problems
    # if has_one_axis(X):
    #     X = [X]
    # if Y is None:
    #     X, Y = [[]] * len(X), X
    if has_one_axis(Y):  # Used to be elif
        Y = [Y]
    # if len(X) != len(Y):
    #     X = X * len(Y)

    kwargs = dict(ax=None, style = ['science',"nature",'no-latex'], save_path = None,
                lgd_title =None, lgd_labels=None,title= None,
                errorbands = ([None]*len(X),[None]*len(X)))
    kwargs.update(pparam)

    # Core plot function
    with plt.style.context(kwargs["style"]):
        print(f"kwargs : {kwargs['ax']}")
        if kwargs["ax"] is not None:
            ax = kwargs["ax"]
        else:
            fig, ax = plt.subplots()    

        for i in range(len(Y)):
            # print(X,"\n",Y)
            print(f"Plotting line {i}")
            ax.plot(X, Y[i], label=kwargs["lgd_labels"][i] if kwargs["lgd_labels"] else i )
            if kwargs["errorbands"][0][i] is not None and kwargs["errorbands"][1][i] is not None:
                #// print(kwargs["errorbands"][0][i], kwargs["errorbands"][1][i])
                ax.fill_between(X, kwargs["errorbands"][0][i], kwargs["errorbands"][1][i], alpha=0.2, lw=0)
 

        if kwargs["ax"] is None:
            ax.legend(title=kwargs["lgd_title"])
            ax.autoscale(tight=True)
            if xyparam: ax.set(**xyparam)
            fig.savefig(f'{kwargs["save_path"]}.svg')
            fig.savefig(f'{kwargs["save_path"]}.jpg', dpi=300)
        


def scatter_lines_plot( X, xylines= None, xyparam= None, style = ['science', "nature",'no-latex'],**pparam): # 'scatter',
    """plot scatter with lines

    Args:
        X (List((x1,y1)...(xn,yn))): _description_
        xylines ([[xs[i],ys[i]] , optional): up-bottom(next line) mapping,
                            xs = [[x1,x2,..xn], [x1,x2,..xn], ...,[[x1,x2,..xn]]], also for ys
                            . Defaults to None.
        xyparam (_type_, optional): _description_. Defaults to None.
        style (list, optional): _description_. Defaults to ['science', "nature",'no-latex'].
    """    
    with plt.style.context(style):
        fig, ax = plt.subplots()
        

        kwargs = dict(save_path = None,lgd_title =None, lgd_labels=None,title= None)
        kwargs.update(pparam)
        
        markers = ['o', 's', '^', 'v', '<', '>', 'd']
        colors = ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e']
        linestyles = ['-', '--', ':', '-.']
        for i in range(len(X)):
            x1,y1 = X[i]
            print(ax)
            ax.scatter(x1, y1,marker=markers[i],label=kwargs["lgd_labels"][i] if kwargs["lgd_labels"] else r"$^\#${}".format(i+1) )


        for i in range(len(xylines)):
            xs, ys = xylines[i]
            lines_plot(xs, ys,**dict(ax = ax, style = ['science',"ieee",'no-latex'],
                        lgd_labels=[kwargs["lgd_labels"][i] if kwargs["lgd_labels"] else r"$^\#${}".format(i+1) ]))

        ax.legend(title=kwargs["lgd_title"])
        ax.autoscale(tight=True)
        if xyparam: ax.set(**xyparam)

    fig.savefig(f'{kwargs["save_path"]}.svg')
    fig.savefig(f'{kwargs["save_path"]}.jpg', dpi=300)






