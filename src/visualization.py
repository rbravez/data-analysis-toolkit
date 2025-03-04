import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def histogram(df, title, variables, xlabel, ylabel, edgecolor='black', bins=30, color=None, density=False, number=1, figsize=(30,18), row=1, column=1):
    """
    Parameters:
    df : DataFrame - The dataset containing the variables
    title : str - Title of the histogram or subplot title
    variables : str or list - Column name(s) to visualize
    xlabel : str - Label for x-axis
    ylabel : str - Label for y-axis
    edgecolor : str - Color of bin edges (default: 'black')
    bins : int - Number of bins (default: 30)
    color : str or None - Color of the bars (default: None, will auto-assign)
    density : bool - Whether to normalize the histogram (default: False)
    number : int - Number of plots (default: 1)
    figsize : tuple - Figure size (default: (10,6))
    row : int - Number of rows for subplot (default: 1)
    column : int - Number of columns for subplot (default: 1)
    """

    # Ensure variables is always a list
    if isinstance(variables, str):
        variables = [variables]

    # Validate that all variables exist in df
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Columns not found in DataFrame: {missing_vars}")

    # If only one plot, use single plot
    if number == 1 or len(variables) == 1:
        plt.figure(figsize=figsize)
        plt.hist(df[variables[0]], bins=bins, edgecolor=edgecolor, color=color, density=density)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    else:
        fig, axes = plt.subplots(row, column, figsize=figsize)
        axes = axes.flatten()  # Ensure axes is always a flat array
        
        for i, var in enumerate(variables):
            if i < len(axes):  # Ensure we do not exceed subplot limit
                axes[i].hist(df[var], bins=bins, edgecolor=edgecolor, color=color, density=density)
                axes[i].set_title(f"{title} - {var}")
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)

        plt.tight_layout()
        plt.show()

def histogram_kde(df, title, variables, xlabel, ylabel, edgecolor='black', bins=30, color=None, kde=False, number=1, figsize=(10,6), row=1, column=1):
    """
    Parameters:
    df : DataFrame - The dataset containing the variables
    title : str - Title of the histogram or subplot title
    variables : list - List of column names to visualize
    xlabel : str - Label for x-axis
    ylabel : str - Label for y-axis
    edgecolor : str - Color of bin edges (default: 'black')
    bins : int - Number of bins (default: 30)
    color : str or None - Color of the bars (default: None, will auto-assign)
    density : bool - Whether to normalize the histogram (default: False)
    number : int - Number of plots (default: 1)
    figsize : tuple - Figure size (default: (10,6))
    row : int - Number of rows for subplot (default: 1)
    column : int - Number of columns for subplot (default: 1)
    """
    
    if number == 1:
        plt.figure(figsize=figsize)
        sns.histplot(df[variables[0]], bins=bins, edgecolor=edgecolor, color=color, kde=kde)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    else:
        fig, axes = plt.subplots(row, column, figsize=figsize)
        axes = axes.flatten()  # Flatten in case of multi-row/column layout
        
        for i, var in enumerate(variables):
            if i < number:  # Ensure we do not exceed the available axes
                sns.histplot(df[var], bins=bins, edgecolor=edgecolor, color=color, kde=kde, ax=axes[i])
                axes[i].set_title(f"{title} - {var}")
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
        
        plt.tight_layout()
        plt.show()

def boxplot(df, title, variables, xlabel, ylabel, color=None, number=1, figsize=(10,6), row=1, column=1):
    """
    Generate one or multiple boxplots from a DataFrame.
    
    Parameters:
    df : DataFrame - The dataset containing the variables.
    title : str - The title of the boxplot or subplot title.
    variables : list - List of column names to visualize.
    xlabel : str - Label for x-axis.
    ylabel : str - Label for y-axis.
    color : str or None - Color of the boxes (default: None, will auto-assign).
    number : int - Number of plots (default: 1).
    figsize : tuple - Figure size (default: (10,6)).
    row : int - Number of rows for subplot (default: 1).
    column : int - Number of columns for subplot (default: 1).
    
    This function allows plotting multiple boxplots side by side by specifying multiple variables.
    """
    fig, axes = plt.subplots(row, column, figsize=figsize)
    
    if number == 1:
        sns.boxplot(y=df[variables[0]], color=color, ax=axes)
        axes.set_title(f"{title} - {variables[0]}")
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for i, var in enumerate(variables):
            if i < number:  # Ensure we do not exceed the available axes
                sns.boxplot(y=df[var], color=color, ax=axes[i])
                axes[i].set_title(f"{title} - {var}")
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
    
    plt.tight_layout()
    plt.show()

def scatterplot(df, title, x_vars, y_vars, hue=None, style=None, size=None, figsize=(10,6)):
    """
    Generate one scatter plot from a DataFrame.
    
    Parameters:
    df : DataFrame - The dataset containing the variables.
    title : str - Title of the scatter plot.
    x_vars : list - List of column names for x-axis.
    y_vars : list - List of column names for y-axis.
    hue : str or None - Column name for color differentiation (categorical variable).
    style : str or None - Column name for style differentiation (e.g., marker type).
    size : str or None - Column name for point size differentiation.
    figsize : tuple - Figure size (default: (10,6)).
    
    This function allows plotting multiple scatter plots by specifying multiple variables.
    """
    plt.figure(figsize=figsize)
    for x_var, y_var in zip(x_vars, y_vars):
        sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue, style=style, size=size)
    plt.title(title)
    plt.xlabel(', '.join(x_vars))
    plt.ylabel(', '.join(y_vars))
    plt.show()

def lineplot(df, title, x_vars, y_vars, hue=None, style=None, figsize=(10,6)):
    """
    Generate one or multiple line plots from a DataFrame.
    
    Parameters:
    df : DataFrame - The dataset containing the variables.
    title : str - Title of the line plot.
    x_vars : list - List of column names for x-axis.
    y_vars : list - List of column names for y-axis.
    hue : str or None - Column name for color differentiation (categorical variable).
    style : str or None - Column name for style differentiation (e.g., line style).
    figsize : tuple - Figure size (default: (10,6)).
    
    This function allows plotting multiple line plots by specifying multiple variables.
    """
    plt.figure(figsize=figsize)
    for x_var, y_var in zip(x_vars, y_vars):
        sns.lineplot(data=df, x=x_var, y=y_var, hue=hue, style=style)
    plt.title(title)
    plt.xlabel(', '.join(x_vars))
    plt.ylabel(', '.join(y_vars))
    plt.show()
    """
    Parameters:
    df : DataFrame - The dataset containing the variables
    title : str - Title of the line plot
    x_var : str - Column name for x-axis
    y_var : str - Column name for y-axis
    hue : str or None - Column name for color differentiation (categorical variable)
    style : str or None - Column name for style differentiation (e.g., line style)
    figsize : tuple - Figure size (default: (10,6))
    """
    plt.figure(figsize=figsize)
    sns.lineplot(data=df, x=x_var, y=y_var, hue=hue, style=style)
    plt.title(title)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()
