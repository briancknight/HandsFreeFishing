import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

def main():
    dir = '/Users/brknight/Documents/GitHub/HandsFreeFishing/measurements/example_fish'
    df = pd.read_excel(os.path.join(dir,'output.xlsx'),sheet_name="Sheet1")
    
    weights = np.array(df['Weight(g)'])
    scales=np.array(df['pred scale'])
    areas=np.array(df['pred area'])
    no_fin_areas=np.array(df['pred area (no fins)'])
    
    densities1 = weights/no_fin_areas
    densities2 = weights/areas
    

    slope, intercept, r, p, se = linregress(no_fin_areas[1::2], weights[1::2])

    print('slope of linear regression yields density estimate: ', slope)
    
    pred_weights = no_fin_areas[::2] * slope
    
    errors = np.abs(pred_weights - weights[::2])

    print('mean error on remaining data is ', np.mean(errors))
    
    # plt.plot(errors)
    # plt.show()
    
    
    slope, intercept, r, p, se = linregress(areas[1::2], weights[1::2])

    print('slope of linear regression yields density estimate: ', slope)
    
    pred_weights = areas[::2] * slope
    
    errors = np.abs(pred_weights - weights[::2])

    print('mean error on remaining data is ', np.mean(errors))
    
    # plt.plot(errors)
    # plt.show()
    
    # fig, ax=plt.subplots()
    # ax.scatter(no_fin_areas, weights)
    # ax.scatter(areas, weights)

    # # Adjust layout to prevent overlap
    # plt.tight_layout()

    # # Show the plot
    # plt.show()
    
if __name__ == "__main__":
    main()