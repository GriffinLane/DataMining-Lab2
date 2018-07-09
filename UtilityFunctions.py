import itertools
from sklearn.metrics import (mean_squared_error, log_loss, accuracy_score,
                            classification_report,accuracy_score,confusion_matrix,
                            roc_auc_score)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def casual_metrics(name, y_test, y_hat, y_proba=None):
    _mse = mean_squared_error(y_test, y_hat)
    _rmse = np.sqrt(_mse)
    _log_loss = log_loss(y_test, y_hat, eps=1e-15, normalize=True, sample_weight=None, labels=None)
    
    print(
        """
        {0}:
            MSE:        {1}
            RMSE:       {2}
            Log Loss:   {3}
        """.format(name, _mse, _rmse, _log_loss)
    )
    
    print(classification_report(y_test, y_hat))
    if y_proba is not None:
        _roc = roc_auc_score(y_test, y_proba)
        print("ROC AUC: ", _roc)
    print("accuracy:  ",accuracy_score(y_test, y_hat))
        
    
    
def conf_pixel_plot(y_test, y_hat, classes, normalize, title="Confusion Matrix", cmap=plt.cm.Reds):
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_hat)
    np.set_printoptions(precision=2)
    
    cm = cnf_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
cmap_center = 0

def custom_color_mapper(name = "custom", val_range = (1.96,1.96), colors = "RdGnBu"):
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(name,colors=colors)
    
    min, max = val_range
    step = max/10.0
    Z = [min,0],[0,max]
    levels = np.arange(min,max+step,step)
    cust_map = plt.contourf(Z, 100, cmap=custom_cmap)#cmap = custom_cmap
    plt.clf()
    return cust_map


def colormap_bar(data, measure_name, y_axis_name, cmap = "seismic", cmap_center = 0,
                 size = [5,15], formatter = "normal", value_format="", title="", **kwargs):
    
    plt.rcParams["figure.figsize"] = size
    with sns.plotting_context(font_scale=5.5):
        if formatter != "normal":
            if formatter is "percent":
                formatter = FuncFormatter(lambda x, pos: '{:.0%}'.format(x))
        else:
            formatter = None
                
        
        data = data.rename(columns={"index":y_axis_name})
        data.sort_values(by=[measure_name])        
        
        piv = pd.pivot_table(data, values=measure_name, index=y_axis_name)
         
        cbar = plt.colorbar( mappable=cmap)   
        ax = sns.heatmap(piv, square=False, cmap=custom.cmap, cbar = False,
                         vmin=0,vmax=1, center=cmap_center,fmt=value_format,
                         cbar_kws={'format': formatter}, annot=True, **kwargs)

        ax.set_xlabel("",fontsize=30)
        ax.set_ylabel(y_axis_name,fontsize=20)
        ax.tick_params(labelsize=13)
        ax.tick_params(axis="x", labelsize = 20)
        ax.set_title(label=title+"\n")
        ax.title.set_size(30)
    
    plt.rcParams["figure.figsize"] = size
    return plt
    