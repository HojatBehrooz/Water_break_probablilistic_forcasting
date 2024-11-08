# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:38:12 2024

@author: hbehrooz
"""
#import os
#import warnings

"""
To use the import lightning.x.y import, you need to install the lightning package: pip install lightning
If you want to use the import pytorch_lightning.x.y style, you need to install pip install pytorch-lightning

You can't mix and match them. Our documentation exclusively uses the new package imports with lightning. That's the new and recommended way.
"""
#this file is the main modelig file that use pipe data and train a model for predicting a break in 2023 pipes
#it use gluonts ackage for trainig a DeepAR model
#import pytorch_lightning as pl
#from lightning.pytorch.callbacks import EarlyStopping
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset


#from gluonts.dataset.common import ListDataset
#from gluonts.time_feature import TimeFeature
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import torch

import torch
#import torch.nn.functional as F
from torch.distributions import Bernoulli as TorchBernoulli
from gluonts.torch.distributions import DistributionOutput
from typing import Dict, Optional, Tuple
#this Bernouli binary distribution which is used for feeding to model 
#the model need to have distribution for porbability forcasting. orginally the Gluonts does not have Beroulli distribution and I creat it by myself
class BernoulliOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"probs": 1}  # Bernoulli only requires the probability of success
    distr_cls: type = TorchBernoulli

    @classmethod
    def domain_map(cls, probs: torch.Tensor):  # type: ignore
        # Bernoulli probabilities must be in the range (0, 1)
        probs = torch.sigmoid(probs)  # Ensures values are in the (0, 1) range
        return (probs.squeeze(-1),)

    @property
    def event_shape(self) -> torch.Size:
        # Bernoulli distribution has scalar events
        return torch.Size([])

    def distribution(self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None):
        (probs,) = distr_args
        return self.distr_cls(probs=probs)

    def loss(self, target: torch.Tensor, distr_args: Tuple[torch.Tensor, ...], loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        distribution = self.distribution(distr_args, loc=loc, scale=scale)
        nll = -distribution.log_prob(target)  # Negative log-likelihood
        return nll
#%%I tried to use and scalled version of Bernouli distribution not help to improve prediction to highier values
# I had a probalm that the higihest probalbility was around 50% not solving with this approch

class ScaledBernoulliOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"probs": 1}  # Bernoulli only requires the probability of success
    distr_cls: type = TorchBernoulli

    def __init__(self, scale: float = 1., **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    @classmethod
    def domain_map(cls, probs: torch.Tensor, scale: float = 1.0):  # type: ignore
        # Bernoulli probabilities must be in the range (0, 1)
        probs = torch.sigmoid(probs)  # Ensures values are in the (0, 1) range
        # Apply scaling factor
        scaled_probs = probs * scale
        return (scaled_probs.squeeze(-1),)

    @property
    def event_shape(self) -> torch.Size:
        # Bernoulli distribution has scalar events
        return torch.Size([])

    def distribution(self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None):
        (probs,) = distr_args
        return self.distr_cls(probs=probs)

    def loss(self, target: torch.Tensor, distr_args: Tuple[torch.Tensor, ...], loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        distribution = self.distribution(distr_args, loc=loc, scale=scale)
        nll = -distribution.log_prob(target)  # Negative log-likelihood
        return nll

#%%read the dataset

"""
['year', 'PIPE_ID', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'S', 'MATERIAL',
       'INSTALLED_YEAR', 'DIAM', 'TYPE', 'LENGTH', 'ZONE', 'X', 'Y', 'BREAKED',
       'BREAKS', 'TSLF', 'avg_temp', 'std_tmp', 'HDD', 'CDD', 'precipitation',
       'rain', 'snow', 'age'],
      dtype='object')
"""
df_file="breakdfv1.csv"
datatype={'year':'int', 'PIPE_ID':'int', 'A':'category', 'B':'category', 'C':'category',
          'D':'category', 'E':'category', 'F':'category', 'G':'category', 'S':'category','MATERIAL':'category',
          'INSTALLED_YEAR':'int', 'DIAM':'int', 'TYPE':'category',
          'LENGTH':'float32', 'ZONE':'category','BREAKED':'int','BREAKS':'int',
          'X':'float32', 'Y':'float32', 'TSLF':np.int32, 'avg_temp':np.float64,
          'std_tmp':np.float64, 'HDD':np.float64,
          'CDD':np.float64, 'precipitation':np.float64, 'rain':np.float64, 'snow':np.float64,'age':'int'}
data_orig=pd.read_csv(df_file,dtype=datatype)
data=data_orig[data_orig['year']<data_orig['year'].max()].copy()
age=data['INSTALLED_YEAR']-data['year']
age[age<0]=0
data['age']=age 
#Make number of pipe a multiple of batchsize =128
#data=data[data['PIPE_ID']<100000].copy()
aa=data['PIPE_ID'].unique()
#data=data[data['PIPE_ID'].isin(aa[:((len(aa)//1280)*1280)])].copy()
data['PIPE_ID']=data['PIPE_ID'].apply(str)
data['PIPE_ID']=data['PIPE_ID'].astype('category')
#%%defin break filed which show at least was one break during a year (1) or not (0))
data['break']=(data['BREAKED']>0).astype(float)
data.drop(columns=['BREAKED'],inplace=True)
#normalizing the numnerical features
normalizing_features=[ 'INSTALLED_YEAR', 'DIAM', 'LENGTH','BREAKS','X', 'Y', 'TSLF', 'avg_temp',
          'std_tmp' ,'HDD','CDD', 'precipitation', 'rain', 'snow','age']
data['year']=data['year'].apply(lambda x: pd.to_datetime(str(x), format='%Y'))

#%%standardize input data (not used)
"""
standradization not improve the output. 
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[normalizing_features] = scaler.fit_transform(data[normalizing_features])
#%%# minmax normalization
#data["time_idx"] = data["year"]
#data["time_idx"] -= data["time_idx"].min()
#add a field as targe shows if ther is more than zero break in year
data_normal=data.copy()
data_normal[normalizing_features] = (data[normalizing_features] - data[normalizing_features].min()) / (data[normalizing_features].max() - data[normalizing_features].min())


#%%resampling for reducing inbalancing
#resample for group of pipes with 0, 1, 2 or 3, and more than 3 break in their life time and resample from each pipe groups with
# a fixed number of pipes. the bigest group would be used as base for sampling fom othe groups (over sampling)
# Group by 'PIPE_ID' and sum the 'break' column

data=data_normal
# Create a DataFrame that contains the total number of breaks per pipe
break_counts = data.groupby('PIPE_ID',observed=False)['break'].sum().reset_index()
break_counts.columns = ['PIPE_ID', 'total_breaks']
np.unique(break_counts['total_breaks'],return_counts=True)
# Define the groups based on total breaks

# Define the group labels
group_labels = ['0_break', '1_break','2_3_break', 'more_than_3_break']

# Create a new column in `break_counts` to identify the group
break_counts['group'] = pd.cut(break_counts['total_breaks'],
                               bins=[-1, 0, 1,3,  float('inf')],
                               labels=group_labels)
print(break_counts.groupby('group',observed=True).count())
# Determine the number of samples to take from each group (maximum group size)
min_samples =break_counts['group'].value_counts().max()

# Sample from each group and repeat the PIPE_IDs accordingly for minority groups
sampled_pipes = []
for group in group_labels:
    group_df = break_counts[break_counts['group'] == group]
    sampled_pipes.extend(group_df['PIPE_ID'].tolist())
    if(len(group_df)<min_samples):        
        sampled_group_df = group_df.sample(n=min_samples-len(group_df), 
                                           replace=True, random_state=42)  # Sample with replacement if needed
        sampled_pipes.extend(sampled_group_df['PIPE_ID'].tolist())

# Create a DataFrame with the sampled PIPE_IDs
sampled_pipes_df = pd.DataFrame(sampled_pipes, columns=['PIPE_ID'])
sampled_pipes_df['new_ID']=sampled_pipes_df.index.values
# Merge to get the balanced dataset
balanced_data = pd.merge(data, sampled_pipes_df, on='PIPE_ID')

# Verify the distribution
#print(balanced_data.groupby('PIPE_ID')['break'].sum().value_counts())
balanced_data.rename(columns={'PIPE_ID':'orig_PIPE_ID'},inplace=True)
balanced_data.rename(columns={'new_ID':'PIPE_ID'},inplace=True)
balanced_data['orig_break']=balanced_data['break']
id_dic=dict(zip(balanced_data['PIPE_ID'],balanced_data['orig_PIPE_ID']))
first_years = data[data['age'] == 0].groupby('PIPE_ID')['year'].min().reset_index()
balanced_data_copy=balanced_data.copy()
#first year of data available for each pipe
#%%
outf=open("output.txt","w")
for predicting_year in range(2019,2024,1):
    data=balanced_data
    #the prediction will be for the last year which is 2023
    #but we can filter data to be untill 2022,2021,2020, to have prediction for 2022,2021,2020 base on historical data
    #here we filter the data for trainig diffrent model and have predited year as:
    #predicting_year=2019 # change if we need to predict other years
    data=data[data['year'].dt.year<=predicting_year].copy()
        #data=filtered_data
    #%%this part create an under sampling and delete the record before installation of each pipe not used
    """# Resampling for reducing imbalance via undersampling
    #data = data_normal
    
    # Create a DataFrame that contains the total number of breaks per pipe
    break_counts = data.groupby('PIPE_ID', observed=False)['break'].sum().reset_index()
    break_counts.columns = ['PIPE_ID', 'total_breaks']
    np.unique(break_counts['total_breaks'], return_counts=True)
    
    # Define the groups based on total breaks
    group_labels = ['0_break', '1_break', '2_3_break', 'more_than_3_break']
    
    # Create a new column in `break_counts` to identify the group
    break_counts['group'] = pd.cut(break_counts['total_breaks'],
                                   bins=[-1, 0, 1, 3, float('inf')],
                                   labels=group_labels)
    print(break_counts.groupby('group', observed=True).count())
    
    # Determine the number of samples to take from each group (minimum group size)
    min_samples = 500#break_counts['group'].value_counts().min()
    
    # Sample from each group to reduce to the minimum size
    undersampled_pipes = []
    for group in group_labels:
        group_df = break_counts[break_counts['group'] == group]
        if len(group_df) > min_samples:
            undersampled_group_df = group_df.sample(n=min_samples, 
                                                    random_state=42)  # Sample without replacement
            undersampled_pipes.extend(undersampled_group_df['PIPE_ID'].tolist())
        else:
            undersampled_pipes.extend(group_df['PIPE_ID'].tolist())
    
    # Create a DataFrame with the undersampled PIPE_IDs
    undersampled_pipes_df = pd.DataFrame(undersampled_pipes, columns=['PIPE_ID'])
    undersampled_pipes_df['new_ID'] = undersampled_pipes_df.index.values
    
    # Merge to get the balanced dataset
    balanced_data = pd.merge(data, undersampled_pipes_df, on='PIPE_ID')
    
    # Verify the distribution
    #print(balanced_data.groupby('PIPE_ID')['break'].sum().value_counts())
    balanced_data.rename(columns={'PIPE_ID': 'orig_PIPE_ID'}, inplace=True)
    balanced_data.rename(columns={'new_ID': 'PIPE_ID'}, inplace=True)
    id_dic = dict(zip(balanced_data['PIPE_ID'], balanced_data['orig_PIPE_ID']))
    
    data = balanced_data
    
    # Assuming `data` is your DataFrame with columns: 'PIPE_ID', 'year', 'age'
    
    # Step 1: Find the first year when each pipe has an age of 0
    first_years = data[data['age'] == 0].groupby('PIPE_ID')['year'].min().reset_index()
    first_years.columns = ['PIPE_ID', 'first_year']
    
    # Step 2: Merge this information back with the original data
    data_with_first_years = pd.merge(data, first_years, on='PIPE_ID')
    
    # Step 3: Filter the dataset to include only rows from the first year onwards
    filtered_data = data_with_first_years[data_with_first_years['year'] >= data_with_first_years['first_year']]
    
    # Optionally, drop the 'first_year' column if not needed
    filtered_data = filtered_data.drop(columns='first_year')
    
    # Verify the results
    print(filtered_data.head())
    print(filtered_data.groupby('PIPE_ID')['year'].min().reset_index())
    filtered_data.to_csv('filtered500.csv',index=False)
    # First year of data available for each pipe
    first_years = data[data['age'] == 0].groupby('PIPE_ID')['year'].min().reset_index()
    """
    #%% This part creat custome dataset for gluonts according to their documentation
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    num_series=len(data.PIPE_ID.unique()) #number of time series are equall to number of pipes
    num_steps= data['year'].max().year-data['year'].min().year+1 #number of avaiable pipe data in year
    start_year=data['year'].min() #start of timeseris
    prediction_len=1
    context_len=12 
    test_len=prediction_len+context_len
    start_test=data['year'].max().year-test_len+1
    custom_ds_metadata={
        "num_seris":num_series,
        "num_steps":num_steps,
        "prediction_length":prediction_len,
        "freq":"1Y",
        "context_length":context_len,
    #    "start":[pd.Period(first_years['year'][k], freq="1Y") for k in range(num_series)] }
        "start":[pd.Period(start_year, freq="1Y") for k in range(num_series)] ,
        "start_test":[pd.Period(start_test, freq="1Y") for k in range(num_series)] }
    #target value break in amatrix rows are pipes and columns time periods
    ########################I SHOULD SETUP DATASET as follow
    #https://ts.gluon.ai/stable/tutorials/forecasting/extended_tutorial.html
    #target value
    target = data.pivot(index='PIPE_ID', columns='year', values='break')
    
    year_list=target.columns.year
    target=target.values
    target_orig=data.pivot(index='PIPE_ID', columns='year', values='orig_break').values
    item_id=data.pivot(index='PIPE_ID', columns='year', values='break').index.values.astype(str).tolist()
    #PAST_FEAT_DYNAMIC_REAL
    avg_temp=data.pivot(index='PIPE_ID', columns='year', values='avg_temp').values
    std_tmp=data.pivot(index='PIPE_ID', columns='year', values='std_tmp').values
    HDD=data.pivot(index='PIPE_ID', columns='year', values='HDD').values
    CDD=data.pivot(index='PIPE_ID', columns='year', values='CDD').values
    precipitation=data.pivot(index='PIPE_ID', columns='year', values='precipitation').values
    snow=data.pivot(index='PIPE_ID', columns='year', values='snow').values
    rain=data.pivot(index='PIPE_ID', columns='year', values='rain').values
    BREAKS=data.pivot(index='PIPE_ID', columns='year', values='BREAKS').values
    TSLF=data.pivot(index='PIPE_ID', columns='year', values='TSLF').values
    
    #FEAT_STATIC_REAL
    INSTALLED_YEAR=data.pivot(index='PIPE_ID', columns='year', values='INSTALLED_YEAR').values[:,0]
    DIAM=data.pivot(index='PIPE_ID', columns='year', values='DIAM').values[:,0]
    LENGTH=data.pivot(index='PIPE_ID', columns='year', values='LENGTH').values[:,0]
    X=data.pivot(index='PIPE_ID', columns='year', values='X').values[:,0]
    Y=data.pivot(index='PIPE_ID', columns='year', values='Y').values[:,0]
    
    #FEAT_STATIC_CAT
    MATERIAL=data.pivot(index='PIPE_ID', columns='year', values='MATERIAL').values[:,0]
    TYPE=data.pivot(index='PIPE_ID', columns='year', values='TYPE').values[:,0]
    
    #PAST_FEAT_DYNAMIC_CAT
    ZONE=data.pivot(index='PIPE_ID', columns='year', values='ZONE').values
    A=data.pivot(index='PIPE_ID', columns='year', values='A').values
    B=data.pivot(index='PIPE_ID', columns='year', values='B').values
    C=data.pivot(index='PIPE_ID', columns='year', values='C').values
    D=data.pivot(index='PIPE_ID', columns='year', values='D').values
    E=data.pivot(index='PIPE_ID', columns='year', values='E').values
    F=data.pivot(index='PIPE_ID', columns='year', values='F').values
    G=data.pivot(index='PIPE_ID', columns='year', values='G').values
    S=data.pivot(index='PIPE_ID', columns='year', values='S').values
    
    train_ds = ListDataset(
        [
            {
                FieldName.ITEM_ID: item_id,
                FieldName.TARGET: target,
                FieldName.START: start,#start_year,###############################################
    #            FieldName.FEAT_DYNAMIC_REAL: ,
                FieldName.PAST_FEAT_DYNAMIC_REAL :[avg_temp,std_tmp,HDD,CDD, 
                                                   precipitation, rain, snow,BREAKS,TSLF],
                FieldName.PAST_FEAT_DYNAMIC_CAT :[ZONE,A,B,C,D,E,F,G,S],
                FieldName.FEAT_STATIC_CAT: [MATERIAL,TYPE],
                FieldName.FEAT_STATIC_REAL: [INSTALLED_YEAR, DIAM, LENGTH,X, Y],
                
            }
            for (item_id,target, start, avg_temp,std_tmp,HDD,CDD, 
                 precipitation, rain, snow,BREAKS,TSLF,
                 ZONE,A,B,C,D,E,F,G,S,
                 MATERIAL,TYPE,
                 INSTALLED_YEAR, DIAM, LENGTH,X, Y) in zip(
                item_id,
                target[:, : -custom_ds_metadata["prediction_length"]],
                custom_ds_metadata["start"],
                avg_temp[:, : -custom_ds_metadata["prediction_length"]],
                std_tmp[:, : -custom_ds_metadata["prediction_length"]],
                HDD[:, : -custom_ds_metadata["prediction_length"]],
                CDD[:, : -custom_ds_metadata["prediction_length"]],
                precipitation[:, : -custom_ds_metadata["prediction_length"]],
                rain[:, : -custom_ds_metadata["prediction_length"]],
                snow[:, : -custom_ds_metadata["prediction_length"]],
                BREAKS[:, : -custom_ds_metadata["prediction_length"]],
                TSLF[:, : -custom_ds_metadata["prediction_length"]],
                
                ZONE[:, : -custom_ds_metadata["prediction_length"]],
                A[:, : -custom_ds_metadata["prediction_length"]],
                B[:, : -custom_ds_metadata["prediction_length"]],
                C[:, : -custom_ds_metadata["prediction_length"]],
                D[:, : -custom_ds_metadata["prediction_length"]],
                E[:, : -custom_ds_metadata["prediction_length"]],
                F[:, : -custom_ds_metadata["prediction_length"]],
                G[:, : -custom_ds_metadata["prediction_length"]],
                S[:, : -custom_ds_metadata["prediction_length"]],
                
                MATERIAL,
                TYPE,INSTALLED_YEAR, DIAM, LENGTH,X, Y
                
                
                
            )
        ],
        freq=custom_ds_metadata["freq"],
    )
                     
    test_ds = ListDataset(
        [
            {
                FieldName.ITEM_ID: item_id,
                FieldName.TARGET: target,
                FieldName.START:start,# start_year,######################################
    #            FieldName.FEAT_DYNAMIC_REAL: ,
                FieldName.PAST_FEAT_DYNAMIC_REAL :[avg_temp,std_tmp,HDD,CDD, 
                                                   precipitation, rain, snow,BREAKS,TSLF],
                FieldName.PAST_FEAT_DYNAMIC_CAT :[ZONE,A,B,C,D,E,F,G,S],
                FieldName.FEAT_STATIC_CAT: [MATERIAL,TYPE],
                FieldName.FEAT_STATIC_REAL: [INSTALLED_YEAR, DIAM, LENGTH,X, Y],
                
            }
            for (item_id,target, start, avg_temp,std_tmp,HDD,CDD, 
                 precipitation, rain, snow,BREAKS,TSLF,
                 ZONE,A,B,C,D,E,F,G,S,
                 MATERIAL,TYPE,
                 INSTALLED_YEAR, DIAM, LENGTH,X, Y) in zip(
                item_id,
                target,
                custom_ds_metadata["start"],
                avg_temp,
                std_tmp,
                HDD,
                CDD,
                precipitation,
                rain,
                snow,
                BREAKS,
                TSLF,
                
                ZONE,
                A,
                B,
                C,
                D,
                E,
                F,
                G,
                S,
                
                MATERIAL,
                TYPE,INSTALLED_YEAR, DIAM, LENGTH,X, Y
                
                
                
            )
        ],
        freq=custom_ds_metadata["freq"],
    )                 
    
    
    
    #%%define the model and trian the model
    #from gluonts.torch.distributions import NormalOutput,BetaOutput,NegativeBinomialOutput,discrete_distribution
    #import lightning.pytorch as pl
    from gluonts.torch.model.deepar import DeepAREstimator
    #from gluonts.mx.trainer.callback import TrainingHistory
    
    from gluonts.evaluation.backtest import make_evaluation_predictions
    #history = TrainingHistory()
    
    estimator = DeepAREstimator(freq='Y',     prediction_length=custom_ds_metadata["prediction_length"],
                                context_length= custom_ds_metadata["context_length"],
                                num_layers=5, hidden_size =1024, lr =1e-5,
                                dropout_rate=0.3,distr_output=BernoulliOutput(),
                                 trainer_kwargs={'accelerator': 'auto', 'max_epochs':20})
    #balanced data: 8,5,1024: 0.0013 after 24 epoch
    predictor = estimator.train(train_ds, num_workers=2)
    #%%make an estimation evaluation according to documentation not used
    #prediction
    # Optionally, perform backtesting to evaluate the model
    # forecast_it, ts_it = make_evaluation_predictions(
    #     dataset=test_ds,
    #     predictor=predictor,
    #     num_samples=100,
    # )
    # # Convert the iterator to a list to access the forecasts
    # forecasts = list(forecast_it)
    # actuals = list(ts_it)
    
    # #poit forcast
    # from gluonts.evaluation import Evaluator
    
    # #probability forcast
    # #from gluonts.evaluation import MultivariateEvaluator
    
    
    # # Assuming you have already generated forecasts and actuals
    # evaluator = Evaluator(quantiles=[0.75,0.8, 0.85, 0.9])  # specify quantiles if needed
    # agg_metrics, item_metrics = evaluator(iter(actuals), iter(forecasts), num_series=len(test_ds))
    
    # #print("Aggregated Metrics:")
    # #print(agg_metrics)
    
    # #print("\nItem Metrics (per time series):")
    # #print(item_metrics)
    # item_metrics['actual']=target[:,-1]
    #%%
    
    
    pred = list(predictor.predict(test_ds))
    #%% transform prediction innto a datset with the orginal value for comparison (all_preds)
    #pred=predictor.predict(test_ds)
    all_preds = list()
    s_d=[]
    i=0
    for item in pred:
        #print(item)
        family = int(item.item_id)
        p = item.samples.mean(axis=0)
        pmedian=item.median
        s_d.append(item.start_date)
        #pred[i,:]=p
        #actual[i,:]=target[int(family),-len(p):]
        i+=1
        #print(i)
        # start_year=item.start_date.to_timestamp().year
        # dates=[pd.to_datetime(str(x), format='%Y') for x in range(start_year,start_year+len(p))]
        #year_range = pd.date_range(start=item.start_date.to_timestamp(), periods=len(p), freq='Y')
        #dates = pd.date_range(start=year_range, periods=1, freq='AS')
        # family_pred = pd.DataFrame({ 'PIPE_ID': family, 'p1': p[0],'a1':target[int(family),-len(p)],
        #                             'p2': p[1],'a2':target[int(family),-len(p)+1],
        #                             'p3': p[2],'a3':target[int(family),-len(p)+2]}, index=[0])
    
        family_pred = pd.DataFrame({ 'PIPE_ID': family, 'predict': p,'actual':target_orig[int(family),-1]})
        all_preds += [family_pred]
    all_preds = pd.concat(all_preds, ignore_index=True)
    all_preds['orig_PIPE_ID']=all_preds['PIPE_ID'].astype(int).map(id_dic)
    #merge_preds = all_preds.merge(test, on=['year', 'PIPE_ID'],how='inner')
    all_preds=all_preds.sort_values(by=['orig_PIPE_ID'])
    all_preds=all_preds.drop_duplicates(subset=['orig_PIPE_ID'],keep='last')
    y_true=all_preds['actual']
    y_prob=all_preds['predict']
    # all_preds['scaled_pred']=all_preds['pred']+(1-all_preds['pred'].max())
    #%%various metrics for evaluation of the prediction accuracy
    from sklearn.metrics import brier_score_loss
    """
    Brier Score
    Definition: Measures the mean squared difference between predicted probabilities and the actual binary outcomes.
    Range: 0 (perfect accuracy) to 1 (worst accuracy).
    Interpretation: Lower Brier score indicates better accuracy of predicted probabilities.
    """
    # Assuming y_true and y_prob are the actual binary outcomes and predicted probabilities
    brier_score = brier_score_loss(y_true, y_prob)
    print("Brier score loss:",np.round(brier_score,4),file=outf)
    
    """
    Precision-Recall Curve and AUC (Area Under the Curve)
    Definition: The Precision-Recall curve plots precision against recall for different threshold values. The AUC of the Precision-Recall curve summarizes the performance.
    Range: 0 to 1.
    Interpretation: Higher AUC indicates better performance, especially in imbalanced datasets.
    """
    from sklearn.metrics import precision_recall_curve, auc
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    #print(f"precision:{precision}, recall:{recall}")
    print(f"pr_auc:{np.round(pr_auc,4)}",file=outf)
    
    """
    ROC Curve and AUC (Area Under the Curve)
    Definition: The ROC curve plots the true positive rate against the false positive rate at various threshold settings. The AUC measures the area under the ROC curve.
    Range: 0.5 (random guessing) to 1 (perfect classification).
    Interpretation: Higher AUC indicates better model performance in distinguishing between the classes.
    """
    from sklearn.metrics import roc_curve, roc_auc_score,auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score_val = roc_auc_score(y_true, y_prob)
    print(f"auc:{np.round(auc_score_val,4)}",file=outf)
    
    """
    Calibration Curve (Reliability Diagram)
    Definition: A graphical representation of the relationship between predicted probabilities and observed frequencies.
    Interpretation: A well-calibrated model’s curve should be close to the diagonal (45-degree line), indicating predicted probabilities match actual outcomes well.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random performance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {predicting_year} prediction')
    plt.legend(loc="lower right")
    plt.show()
    
    #%%
    #print(np.sum((all_preds['p80']==1)==all_preds['actual'])/np.sum(all_preds['actual']))
    print("from:",np.sum(all_preds['actual']),"failure in:",predicting_year,np.sum(all_preds.loc[all_preds['actual']==1]['predict']<.01), "not predicted in p80",file=outf)
    print("from:",np.sum(all_preds['actual']==0),"no failure in:",predicting_year,np.sum(all_preds.loc[all_preds['actual']==0]['predict']>.5), "not predicted in p80",file=outf)
    from sklearn.metrics import confusion_matrix
    all_preds=all_preds.sort_values('predict',ascending=False).reset_index()
    #percent of pipes nnulay breaks during the study period was 1.76% which is used to find top 
    #list of prioritze (the most probable ) pipe breaking in this year
    priority=int(len(all_preds)*0.0176)
    all_preds['scaled_pred']=0 
    all_preds.loc[:priority,'scaled_pred']=(all_preds.loc[:priority,'predict']>0).astype(int)
    cm = confusion_matrix(all_preds['actual'], all_preds['scaled_pred'])
    
    print("Confusion Matrix:",file=outf)
    print(cm,file=outf)
    
    from sklearn.metrics import ConfusionMatrixDisplay
    
    # Display the confusion matrix based on 1.76% of the top pririty with highest probability of failure as failed 
    #1.76% was average annula failure rate
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fail", "Fail"])
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    all_preds.to_csv(f'prediction{predicting_year}.csv',index=False)
    #%%transfer prediction from all_preds to original dataset for next runing
    # Step 1: Create a dictionary from all_preds with 'orig_PIPE_ID' as keys and 'scaled_pred' as values
    pred_dict = dict(zip(all_preds['orig_PIPE_ID'], all_preds['scaled_pred']))
    
    # Step 2: Filter balanced_data for rows where 'year' is 2019 and update 'BREAKS' using the dictionary
    mask = balanced_data['year'].dt.year == predicting_year
    balanced_data.loc[mask, 'break'] = balanced_data.loc[mask, 'orig_PIPE_ID'].map(pred_dict).fillna(balanced_data['break'])
    print("----------------",file=outf)
outf.close()
#data understanding
#%%various dataset measures for report in paper
#distribution of installed year
installed_year=data_orig.groupby('PIPE_ID').last()['INSTALLED_YEAR'].describe()
diam=data_orig.groupby('PIPE_ID').last()['DIAM'].describe()
materail=data_orig.groupby('PIPE_ID').last()['MATERIAL'].describe()
matrial_type=data_orig.groupby('PIPE_ID').last()['MATERIAL'].unique()
diam=data_orig.groupby('PIPE_ID').last()['DIAM'].describe()

type_p=data_orig.groupby('PIPE_ID').last()['TYPE'].unique()

length_=data_orig.groupby('PIPE_ID').last()['LENGTH'].describe()
zone=data_orig.groupby('PIPE_ID').last()['ZONE'].unique()


plt.hist(installed_year)