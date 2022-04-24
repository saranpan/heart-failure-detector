import plotly.express as px
import opendatasets as od
import json, os
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

def import_dataset(dataset_url):

    with open(r'C:\Users\Wallik\.kaggle\kaggle.json','r') as f:
        code = json.load(f)

    os.environ['KAGGLE_USERNAME'] = code['username']
    os.environ['KAGGLE_KEY'] = code['key']

    od.download(dataset_url, data_dir='.')

def cat_num_feature_seperator(df):
    idx = df.applymap(lambda x: isinstance(x, (str,bool))).all(0)
    cat_df = df[df.columns[idx]]
    num_df = df[df.columns[~idx]]

    return cat_df , num_df


class statistical_inference:
    def __init__(self,df,feature_1,feature_2,alternative = 'two-sided',significance = 0.025):
        self.df = df

        self.feature_1 = feature_1
        self.feature_2 = feature_2

        self.alternative = alternative
        self.significance = significance

        # gdf : grouped df    
        self.gdf = df.groupby([feature_1, feature_2]).age.agg(['count'])
        self.gdf['total'] = self.gdf.groupby(feature_1)['count'].transform('sum')
        
    
    def two_proportion_inference_plot(self):
        
        # rgdf : reset gdf
        rgdf = self.gdf.reset_index()
        fig = px.bar(rgdf, x = self.feature_1, y = 'count',
                    color = self.feature_2,
                    text_auto=True,
                    barmode='group',
                    title = f'A Comparison of {self.feature_1} in which distinguished by {self.feature_2}'
                    )

        return fig
    
    def two_proportion_inference(self):

        # Temporary : We got special case where the categorical is not boolean
        
        label = True
        
        if self.feature_2 == 'sex':
          label = 'Male'

        #sample 1
        k1 = self.gdf.query(f'{self.feature_2} == @label')['count'].iloc[0]
        n1 = self.gdf.groupby(self.feature_1)['total'].mean()[0]

        #sample 2
        k2 = self.gdf.query(f'{self.feature_2} == @label')['count'].iloc[1]
        n2 = self.gdf.groupby(self.feature_1)['total'].mean()[1]

        #gather
        successes = np.array([k1, k2])
        samples = np.array([n1, n2])

        # Test Stat starts here
        stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative=self.alternative)

        if p_value > self.significance:
            conclusion = ":no_good: We fail to reject the null hypothesis - we have nothing else to say"

        else:
            conclusion = ":ok_woman: We reject the null hypothesis - suggest the alternative hypothesis is true"
        
        return stat,p_value,conclusion