import numpy as np
import rpy2.robjects as robjects
import pandas as pd

def load_data():

    #Load the RData file
    robjects.r['load']('Wine.RData')
    Wine = robjects.globalenv['Wine']

    #Extract x_learning:
    x_learning = Wine.rx2('x.learning')
    x_learning = pd.DataFrame(np.array(x_learning).reshape((-1, 256)))

    #Extract x_test:
    x_test = Wine.rx2('x.test')
    x_test = pd.DataFrame(np.array(x_test).reshape((-1,256)))
    return x_learning, x_test


