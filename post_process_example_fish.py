import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress


def main():
    dir = os.path.join('measurements','example_fish')
    df = pd.read_excel(os.path.join(dir,'output.xlsx'),sheet_name=0)
    
    save_figures=False
    
    weights = np.array(df['Weight(g)'])
    scales=np.array(df['pred scale'])
    areas=np.array(df['pred area'])
    no_fin_areas=np.array(df['pred area (no fins)'])
    
    densities1 = weights/no_fin_areas
    densities2 = weights/areas
    
    with_no_fin_area = True
    with_fin_area = False
    
    if with_no_fin_area:
        slope, intercept, r, p, se = linregress(no_fin_areas[::2], weights[::2])

        print('slope of linear regression yields density estimate: ', slope, ' grams per pixel')
        print('r val = ', r)
        print('p val = ', p)
        print('se val = ', se)
        print('intercept = ', intercept)
        
        # TRAINING: 
        
        pred_weights = no_fin_areas[::2] * slope + intercept
        
        print('ground truth weights: \n', weights[::2], '\n\n')
        print('predicted weights: \n', np.round(pred_weights,2), '\n\n')
        
        train_errors = np.abs(pred_weights - weights[::2])
        train_mae = np.mean(train_errors)
        train_mape = train_mae/np.mean(weights[::2])*100.0
        
        print('mean error (training) ', train_mae, '\n')   
        print('mean percentage error (training) ', train_mape)
        
        # TESTING
        
        pred_weights = no_fin_areas[1::2] * slope + intercept
        
        print('ground truth weights: \n', weights[1::2], '\n\n')
        print('predicted weights: \n', np.round(pred_weights,2), '\n\n')
        
        test_errors = np.abs(pred_weights - weights[1::2])
        test_mae = np.mean(test_errors)
        test_mape = test_mae/np.mean(weights[1::2])*100.0
        
        print('mean error (testing) ', np.mean(test_errors), '\n')
        print('mean percentage error (testing) ', test_mape)
        
        # plt.plot(errors)
        # plt.show()
        
        # plt.plot(errors)
        # plt.show()
        
        fig, ax=plt.subplots()
        ax.scatter(no_fin_areas[::2], weights[::2])
        ax.scatter(no_fin_areas[1::2], weights[1::2],color='red')
        ax.plot(range(0,500), slope * range(0,500) + intercept, linestyle='-', color='black')
        ax.legend(['training', 'testing', 'regression line'])
        ax.set_ylabel('Weight (g)')
        ax.set_xlabel('(No Fin) Segmentation Area')
        ax.set_title('r^2 = ' + str(np.round(r**2,2)) + 
                    ', MAE (train) =  ' + str(np.round(train_mae,3)) + ', ' + str(np.round(train_mape,2)) + '%' +
                    ', MAE (test) =  ' + str(np.round(test_mae,3)) + ', ' + str(np.round(test_mape,2)) + '%')
        # ax.scatter(areas, weights)
        # plt.plot(np.linspace(0,))

        # Adjust layout to prevent overlap
        plt.tight_layout()
        if save_figures:
            plt.savefig('no_fin_segmentation_area_regression.eps', dpi=200)
            plt.savefig('no_fin_segmentation_area_regression.png', dpi=200)
        # Show the plot
        plt.show()
    
    if with_fin_area:
        # REPEAT INCLUDING FIN DATA
        slope, intercept, r, p, se = linregress(areas[::2], weights[::2])

        print('slope of linear regression yields density estimate: ', slope, ' grams per pixel')
        print('r val = ', r)
        print('p val = ', p)
        print('se val = ', se)
        print('intercept = ', intercept)
        
        # TRAINING: 
        
        pred_weights = areas[::2] * slope + intercept
        
        print('ground truth weights: \n', weights[::2], '\n\n')
        print('predicted weights: \n', np.round(pred_weights,2), '\n\n')
        
        train_errors = np.abs(pred_weights - weights[::2])
        train_mae = np.mean(train_errors)
        train_mape = train_mae/np.mean(weights[::2])*100.0
        
        print('mean error (training) ', train_mae, '\n')   
        print('mean percentage error (training) ', train_mape)
        
        # TESTING
        
        pred_weights = areas[1::2] * slope + intercept
        
        print('ground truth weights: \n', weights[1::2], '\n\n')
        print('predicted weights: \n', np.round(pred_weights,2), '\n\n')
        
        test_errors = np.abs(pred_weights - weights[1::2])
        test_mae = np.mean(test_errors)
        test_mape = test_mae/np.mean(weights[1::2])*100.0
        
        print('mean error (testing) ', np.mean(test_errors), '\n')
        print('mean percentage error (testing) ', test_mape)
        
        # plt.plot(errors)
        # plt.show()
        
        # plt.plot(errors)
        # plt.show()
        
        fig, ax=plt.subplots()
        ax.scatter(areas[::2], weights[::2])
        ax.scatter(areas[1::2], weights[1::2],color='red')
        ax.plot(range(0,600), slope * range(0,600) + intercept, linestyle='-', color='black')
        ax.legend(['training', 'testing', 'regression line'])
        ax.set_ylabel('Weight (g)')
        ax.set_xlabel('Segmentation Area')
        ax.set_title('r^2 = ' + str(np.round(r**2,2)) + 
                    ', MAE (train) =  ' + str(np.round(train_mae,3)) + ', ' + str(np.round(train_mape,2)) + '%' +
                    ', MAE (test) =  ' + str(np.round(test_mae,3)) + ', ' + str(np.round(test_mape,2)) + '%')
        # ax.scatter(areas, weights)
        # plt.plot(np.linspace(0,))

        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        if save_figures:
            plt.savefig('segmentation_area_regression.eps', dpi=200)
            plt.savefig('segmentation_area_regression.png', dpi=200)
        
        # Show the plot
        plt.show()
    
    
def fish_filet():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

    # dir = os.path.join('measurements','example_fish')
    # data = pd.read_excel(os.path.join(dir,'output.xlsx'),sheet_name=0)
    dir = os.path.join('measurements','CABA_Fish')
    data = pd.read_excel(os.path.join(dir,'combined_output.xlsx'),sheet_name=0)

    X = data[['head_half','tail_half', 'pred FL']]

    # Define independent and dependent variables
    X = data[['pred FL', 'pred area (no fins)', 'Sector Area 0', 'Sector Area 1', 
              'Sector Area 2','Sector Area 3', 'Sector Area 4','Sector Area 5',
              'Sector Area 6', 'Sector Area 7', 'Sector Area 8', 'Sector Area 9',
              'Sector Area 10', 'Sector Area 11']]# Select your independent variables
    # X = data[['pred FL', 'pred area (no fins)', 'Sector Area 0', 
    #         'Sector Area 2', 'Sector Area 4', 'Sector Area 6', 
    #         'Sector Area 8','Sector Area 10']]# Select your independent variables
    # X = data[['pred FL', 'pred area (no fins)']]# Select your independent variables
    
    
    X = data[['pred area (no fins)', 'pred FL']]#
    # X = data[['Sector Area 0', 
    #           'Sector Area 2', 'Sector Area 4',
    #           'Sector Area 6',
    #           'Sector Area 8',
    #           'Sector Area 10',]] # Select your independent variables
    y = data['Weight (g)']  # Select your dependent variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)

    # Create a linear regression object
    model = LinearRegression(positive=True)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("GT Weights: \n", np.array(y_test), "\n")
    print("Pred Weights: \n", np.round(y_pred,2), "\n")
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)*100


    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Mean Squared Error:', mse)
    print('Mean Absolute Error:', mae)
    print('Mean Absolute Percentage Error:', mape)
    print('R-squared:', r2)

    # Print the coefficients
    print('Intercept:', model.intercept_)
    print('Coefficients:', model.coef_)
    
if __name__ == "__main__":
    fish_filet()