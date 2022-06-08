# from turtle import color
from d2l import tensorflow as d2l

#! more in d2l: show_images, show_heat_maps,plot,

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """plot img list or array in NXN order

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 
        if isinstance(img, np.ndarray):
            ax.imshow(img)
        else:
            ax.imshow(d2l.numpy(img))

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# default d2l.plot
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None, save_path = None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    d2l.set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    d2l.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    if save_path:
        d2l.plt.savefig(save_path, bbox_inches = 'tight',facecolor="white", transparent=True)




def show_pair_hist(legend, xlabel, ylabel, xlist, ylist):
  
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/') # refer to https://www.kite.com/python/docs/matplotlib.patches.Rectangle.set_hatch
    d2l.plt.legend(legend)


def show_multi_hist(legend, xlabel, ylabel, hist_data_list, title = None, savefig = False, save_path="", bins =None):
    """Plot muliti histograms togethear 
    
        Example:
        e.g https://datavizpyr.com/overlapping-histograms-with-matplotlib-in-python/

        # set seed for reproducing
        np.random.seed(42)
        n, mean_mu1, sd_sigma1 = 5000, 60, 15
        data1 = np.random.normal(mean_mu1, sd_sigma1, n)
        mean_mu2, sd_sigma2 =80, 15
        data2 = np.random.normal(mean_mu2, sd_sigma2, n)
        data3 = data1+data2

        legend = ["data1", "data2", "data3"]
        xlabel, ylabel = "Data", "Count"
        hist_data_list = [data1, data2, data3]
        title = "Multiple Histograms with Matplotlib"

        plot.show_multi_hist(legend, xlabel, ylabel, hist_data_list, title = title, savefig = False, save_path="")

    """
    with plt.style.context(['science' ,'no-latex']):
        d2l.set_figsize(figsize=(8,6))  # 3.5, 2.5
        for i in range(len(hist_data_list)):
            data = hist_data_list[i]
            if bins:
                d2l.plt.hist(data, bins = bins, alpha=0.5)
            else:
                d2l.plt.hist(data, alpha=0.5) # bins=100, , label= legend[i]

        
        d2l.plt.xlabel(xlabel)
        d2l.plt.ylabel(ylabel)    
        d2l.plt.legend(legend)
        d2l.plt.ylim(0,140)

        if title:
            d2l.plt.title(title)
        if savefig:
            d2l.plt.savefig(save_path + "_".join(title.split())+".png") # .svg

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),cmap='magama'): # Reds
    """Show heatmaps of matrices.
    
    Arg
    - matrices:
    _xlabel&ylabel : name in x&y axis

    Defined in :numref:`sec_attention-cues`
    """
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

def show_heatmaps_inline(matrices_list,rows_num, cols_num, camp='viridis',figure_size = (7,7), category_labels=None, save_path =None):
    """plot heatmaps in one row or one col

    Args:
        matrices_list (_type_): _description_
        rows_num (_type_): _description_
        cols_num (_type_): _description_
        camp (str, optional): _description_. Defaults to 'viridis'.
        figure_size (tuple, optional): _description_. Defaults to (7,7).
        category_labels (_type_, optional): _description_. Defaults to None.
        save_path (_type_, optional): _description_. Defaults to None.
    """    
    import d2l.tensorflow as d2l
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors   

    vmin, vmax = np.min(matrices_list), np.max(matrices_list)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(rows_num, cols_num)  # 20 if 9
    d2l.set_figsize(figure_size)
    fig.subplots_adjust(wspace = .1,hspace = 0)

    for i, col in enumerate(axes):
        # for j, col in enumerate(row):
            im = col.pcolormesh(matrices_list[i][::-1,:], norm=norm, cmap=plt.get_cmap(camp)) 

            for pos in ['right',"top","left", "bottom"]:
                col.spines[pos].set_visible(False)  
            if category_labels:
                col.set_xticks(np.arange(len(category_labels))+0.5)
                col.set_yticks(np.arange(len(category_labels))+0.5)
                col.set_xticklabels(labels=category_labels)
                col.set_yticklabels(labels=category_labels[::-1])  # reverse label
                col.tick_params(top=True, bottom=False,
                                labeltop=True, labelbottom=False)
                # Rotate the tick labels and set their alignment.
                plt.setp(col.get_xticklabels(), rotation=-45, ha="right",
                            rotation_mode="anchor")
                col.tick_params(which="minor", bottom=False, left=False)
    plt.subplots_adjust(wspace =0.2, hspace =0)
    # manually change the position of bar cbaxes = fig.add_axes([0.01, 0.01, 0.03, 0.8])  cb = plt.colorbar(im, cax = cbaxes)    
    fig.colorbar(im,ax=axes, location = "right",shrink=0.6) # [ax, bx]  #! col may need to change

    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight',facecolor="white", transparent=True)
    plt.show()

def show_multi_heatmaps(matrices_list,rows_num, cols_num, camp='viridis',figure_size = (7,7), category_labels=None, save_path =None):
    """plot multi heatmaps

    Args:
        matrices_list (_type_): _description_
        rows_num (_type_): _description_
        cols_num (_type_): _description_
        camp (str, optional): _description_. Defaults to 'viridis'.
        figure_size (tuple, optional): _description_. Defaults to (7,7).
        category_labels (_type_, optional): _description_. Defaults to None.
        save_path (_type_, optional): _description_. Defaults to None.
    """    
    import d2l.tensorflow as d2l
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors   

    vmin, vmax = np.min(matrices_list), np.max(matrices_list)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(rows_num, cols_num)  # 20 if 9
    d2l.set_figsize(figure_size)
    fig.subplots_adjust(wspace = .1,hspace = 0)

    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            im = col.pcolormesh(matrices_list[i][::-1,:], norm=norm, cmap=plt.get_cmap(camp)) 

            for pos in ['right',"top","left", "bottom"]:
                col.spines[pos].set_visible(False)  
            if category_labels:
                col.set_xticks(np.arange(len(category_labels))+0.5)
                col.set_yticks(np.arange(len(category_labels))+0.5)
                col.set_xticklabels(labels=category_labels)
                col.set_yticklabels(labels=category_labels[::-1])  # reverse label
                col.tick_params(top=True, bottom=False,
                                labeltop=True, labelbottom=False)
                # Rotate the tick labels and set their alignment.
                plt.setp(col.get_xticklabels(), rotation=-45, ha="right",
                            rotation_mode="anchor")
                col.tick_params(which="minor", bottom=False, left=False)
    plt.subplots_adjust(wspace =0.2, hspace =0)
    
    # manually change the position of bar cbaxes = fig.add_axes([0.01, 0.01, 0.03, 0.8])  cb = plt.colorbar(im, cax = cbaxes)    
    fig.colorbar(im,ax=axes, location = "right",shrink=0.6) # [ax, bx]  #! col may need to change

    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight',facecolor="white", transparent=True)
    plt.show()

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.

    Defined in :numref:`sec_anchor`"""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization.
    Arg:
        - f : function
    Defined in :numref:`subsec_gd-learningrate`"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')