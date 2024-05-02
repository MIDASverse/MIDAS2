from sklearn.linear_model import LinearRegression
import MIDAS2 as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

## Data generation as per King et al (2001)
def MAR1(n: int):
    dta = np.random.multivariate_normal(
        mean = [0,0,0,0,0],
        cov = [
            [1, -.12, -.1, .5, .1,],
            [-.12, 1, .1, -.6, .1,],
            [-.1, .1, 1, -.5, .1,],
            [.5, -.6, -.5, 1, .1,],
            [.1, .1, .1, .1, 1]
        ],
        size = n
    )
    return pd.DataFrame(dta, columns = ['x' + str(i) for i in range(1,6)])

def make_missing(data: pd.DataFrame):
    
    n = data.shape[0]
    M = np.ndarray(data.shape, dtype = bool)
    U1 = np.random.uniform(0,1,n)
    
    
    # Y and X4 are MCAR:
    M[:,0] = U1 < 0.85
    M[:,4] = U1 < 0.85
  
    # X3 is always observed:
    M[:,3] = True
  
    # X1 and X2 are MAR:
    U2 = np.random.uniform(0,1,n)
    M[:,1] = ~np.all([data.iloc[:,3] < -1, U2 < 0.9], axis = 0)
    
    U3 = np.random.uniform(0,1,n)
    M[:,2] = ~np.all([data.iloc[:,3] < -1, U3 < 0.9], axis = 0) 
    
    data[~M] = np.nan
    
    return data

if __name__ == "__main__":
    ## simulation
    B = 1000
    M = 10

    np.random.seed(89)
    torch.manual_seed(89)

    results = pd.DataFrame(columns = ['type','coef1','coef2','coef3','rmse'])

    for b in range(B):
        
        print("Simulation: " + str(b))
        
        train_data = MAR1(1000)
        missing_data = make_missing(train_data.copy())
        
        full_lm = LinearRegression(fit_intercept=True)
        full_lm.fit(train_data.iloc[:,[1,2]], train_data.iloc[:,0])
        full_coefs = [full_lm.intercept_] + full_lm.coef_.tolist()
        full_rmse = np.sqrt(np.mean((full_lm.predict(train_data.iloc[:,[1,2]]) - train_data.iloc[:,0])**2))
        
        results.loc[len(results)] = ['full'] + full_coefs + [full_rmse]
        
        # train imputer
        MIDAS_model = md.MIDAS()
        MIDAS_model.fit(missing_data, epochs = 25, batch_size = 256, lr = 0.01, verbose = False)
        imputed_data = MIDAS_model.transform(m = M)
        
        midas_coefs = None
        midas_rmse = 0
        for imp in imputed_data:
            imp_lm = LinearRegression(fit_intercept=True)
            imp_lm.fit(imp.iloc[:,[1,2]], imp.iloc[:,0])
            imp_coefs = [imp_lm.intercept_] + imp_lm.coef_.tolist()
            if midas_coefs is None:
                midas_coefs = np.array(imp_coefs)
            else:
                midas_coefs += np.array(imp_coefs)
            midas_rmse += np.sqrt(np.mean((imp_lm.predict(imp.iloc[:,[1,2]]) - imp.iloc[:,0])**2))
        
        midas_coefs/= M
        midas_rmse /= M
        
        results.loc[len(results)] = ['midas'] + midas_coefs.tolist() + [midas_rmse]
        
    results_wide = results.melt(id_vars = ['type'], value_vars = ['coef1','coef2','coef3'])
        
    sns.displot(results_wide, x="value", hue="type", kind="kde", col = "variable")
    plt.savefig('../Figures/midas2_torch_mar1.png')
        
    results[['type','rmse']].groupby(['type']).mean()