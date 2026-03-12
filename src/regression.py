import pandas as pd
import statsmodels.formula.api as smf
import itertools

def predict_independent_feats(feats_list:list,df:pd.DataFrame):
    '''
        generates a summary list of single variable OLS
        
    '''
    results = []

    for f in feats_list:
        formula = f'popularity ~ {f}'

        model = smf.ols(formula, data=df).fit()
        
        # edge case cuz explicit turns into a boolean
        if f == 'explicit':
            results.append({
            'Feature': f,
            'Coefficient': model.params['explicit[T.True]'],
            'P-Value': model.pvalues['explicit[T.True]'],
            'R-Squared': model.rsquared
            })
                
        else:
            results.append({
                'Feature': f,
                'Coefficient': model.params[f],
                'P-Value': model.pvalues[f],
                'R-Squared': model.rsquared
            })

    # create table
    if results:
        results_df = pd.DataFrame(results).sort_values(by='R-Squared', ascending=False)
        print(results_df.to_string(index=False))


def brute_force_OLS(feats_list, df, max_combination_size=12):

    '''
        checks the OLS statistics of every combination of features
    '''
    
    results = []

    for k in range(1, max_combination_size + 1):
        for combo in itertools.combinations(feats_list, k):
            
            formula = 'popularity ~ ' + ' + '.join(combo)
            
            model = smf.ols(formula, data=df).fit()
            
            results.append({
                'Num_Features': k,
                'Features': combo,
                'R-Squared': model.rsquared,
                'Adj_R-Squared': model.rsquared_adj,
                'AIC': model.aic,
                'BIC': model.bic
            })

    results_df = pd.DataFrame(results)

    # Sort by Adjusted R² (better metric for comparison)
    results_df = results_df.sort_values(by='Adj_R-Squared', ascending=False)

    print(results_df.head(20).to_string(index=False))