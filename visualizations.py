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
        return fig
    
    # Calculate median momentum for each industry for sorting
    industry_median = data_df.groupby("Industry")["momentum"].median().sort_values(ascending=False)
    sorted_industries = industry_median.index.tolist()
    
    fig = px.box(data_df, x="Industry", y="momentum", 
                 category_orders={"Industry": sorted_industries},
                 color="Industry",
                 title="Momentum Scores by Industry",
                 labels={"momentum": "Momentum Score", "Industry": "Industry"})
    
    fig.update_layout(
        xaxis_title="Industry",
        yaxis_title="Momentum Score",
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis={'categoryorder':'array', 'categoryarray':sorted_industries}
    )
    
    # Adjust for readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_top_bottom_momentum(data_df, n=10):
    """
    Create a bar chart of top and bottom N stocks by momentum
    
    Parameters:
    -----------
    data_df : DataFrame
        DataFrame containing momentum scores and company information
    n : int
        Number of top and bottom stocks to show
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with top and bottom stocks
    """
    if data_df.empty or "momentum" not in data_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No momentum data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sort by momentum and get top and bottom N
    sorted_df = data_df.sort_values("momentum", ascending=False)
    top_n = sorted_df.head(n)
    bottom_n = sorted_df.tail(n)
    
    # Combine with a classification column
    top_n = top_n.assign(classification="Top")
    bottom_n = bottom_n.assign(classification="Bottom")
    plot_df = pd.concat([top_n, bottom_n])
    
    # Create labels for x-axis
    plot_df["label"] = plot_df.apply(
        lambda x: f"{x['symbol']} ({x['Company'][:15]}{'...' if len(x['Company']) > 15 else ''})", 
        axis=1
    )
    
    # Create the plot
    fig = px.bar(plot_df, 
                 x="label", 
                 y="momentum", 
                 color="classification",
                 color_discrete_map={"Top": "#4CAF50", "Bottom": "#F44336"},
                 title=f"Top and Bottom {n} Stocks by Momentum",
                 labels={"momentum": "Momentum Score", "label": "Stock"})
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Momentum Score",
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    # Adjust for readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_industry_breakdown(industry_df):
    """
    Create a pie chart of S&P 500 industry breakdown
    
    Parameters:
    -----------
    industry_df : DataFrame
        DataFrame containing industry counts and percentages
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with industry breakdown pie chart
    """
    if industry_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No industry data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.pie(industry_df, 
                values="Count", 
                names="Industry",
                title="S&P 500 Industry Breakdown",
                hover_data=["Percentage"])
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    fig.update_layout(
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig

def plot_momentum_heatmap(data_df):
    """
    Create a treemap of momentum scores by industry and company
    
    Parameters:
    -----------
    data_df : DataFrame
        DataFrame containing momentum scores and company information
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with momentum heatmap
    """
    if data_df.empty or "momentum" not in data_df.columns or "Industry" not in data_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available for heatmap", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create a treemap
    fig = px.treemap(data_df, 
                    path=[px.Constant("S&P 500"), "Industry", "symbol"],
                    values="factor_rank",
                    color="momentum",
                    color_continuous_scale="RdBu_r",
                    title="Momentum Heat Map by Industry",
                    hover_data=["Company", "momentum"])
    
    fig.update_layout(
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig
