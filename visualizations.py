import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_momentum_distribution(data_df):
    """
    Create a histogram of momentum scores
    
    Parameters:
    -----------
    data_df : DataFrame
        DataFrame containing momentum scores
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with momentum distribution
    """
    if data_df.empty or "momentum" not in data_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No momentum data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.histogram(data_df, x="momentum", 
                       color_discrete_sequence=['#0068c9'],
                       title="Distribution of Momentum Scores",
                       labels={"momentum": "Momentum Score"},
                       nbins=30)
    
    fig.update_layout(
        xaxis_title="Momentum Score",
        yaxis_title="Count",
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig

def plot_industry_momentum(data_df):
    """
    Create a box plot of momentum scores by industry
    
    Parameters:
    -----------
    data_df : DataFrame
        DataFrame containing momentum scores and industry information
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with industry momentum box plot
    """
    if data_df.empty or "momentum" not in data_df.columns or "Industry" not in data_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No industry momentum data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
