import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import calendar
from datetime import datetime

def format_month_and_sort(df, x_axis):
    """Format month numbers as month names and sort data by date if applicable.
    
    Args:
        df: DataFrame containing the data
        x_axis: Column name to use for x-axis
        
    Returns:
        DataFrame with formatted months and sorted by date if applicable
    """
    # Create a copy to avoid modifying the original dataframe
    formatted_df = df.copy()
    
    # Check if the column exists in the dataframe
    if x_axis in formatted_df.columns:
        # Try to detect if column contains month-year format like "Jun 2025", "Jan 2025", etc.
        if formatted_df[x_axis].dtype == 'object':
            # Check if values match month-year pattern
            month_year_pattern = formatted_df[x_axis].str.match(r'^[A-Za-z]{3,9}\s+\d{4}$')
            
            if month_year_pattern.any():
                # Create a new column for sorting
                formatted_df['_sort_key'] = pd.NaT
                
                # Try different date formats for parsing
                for fmt in ['%b %Y', '%B %Y']:
                    # Only try to parse values that haven't been successfully parsed yet
                    mask = formatted_df['_sort_key'].isna()
                    if not mask.any():
                        break
                        
                    try:
                        formatted_df.loc[mask, '_sort_key'] = pd.to_datetime(
                            formatted_df.loc[mask, x_axis], format=fmt, errors='coerce'
                        )
                    except:
                        pass
                
                # If we still have NaT values, try a more flexible approach
                if formatted_df['_sort_key'].isna().any():
                    def parse_month_year(text):
                        if pd.isna(text):
                            return pd.NaT
                            
                        # Extract month and year parts
                        parts = str(text).strip().split()
                        if len(parts) != 2:
                            return pd.NaT
                            
                        month_str, year_str = parts[0], parts[1]
                        
                        # Try to convert month name to month number
                        try:
                            # Try abbreviated month names (Jan, Feb, etc.)
                            month_num = list(calendar.month_abbr).index(month_str[:3].title())
                        except ValueError:
                            try:
                                # Try full month names (January, February, etc.)
                                month_num = list(calendar.month_name).index(month_str.title())
                            except ValueError:
                                return pd.NaT
                        
                        # Try to convert year string to integer
                        try:
                            year_num = int(year_str)
                        except ValueError:
                            return pd.NaT
                            
                        # Create datetime object
                        try:
                            return pd.Timestamp(year=year_num, month=month_num, day=1)
                        except:
                            return pd.NaT
                    
                    # Apply custom parsing function to rows that still have NaT
                    mask = formatted_df['_sort_key'].isna()
                    if mask.any():
                        formatted_df.loc[mask, '_sort_key'] = formatted_df.loc[mask, x_axis].apply(parse_month_year)
                
                # Sort by the datetime column
                formatted_df = formatted_df.sort_values('_sort_key')
                
                # Drop the temporary sorting column
                formatted_df.drop('_sort_key', axis=1, inplace=True)
                
                return formatted_df
            
            # Check if column contains year-month format (YYYY-MM or similar)
            year_month_pattern = formatted_df[x_axis].str.match(r'^\d{4}[-/]\d{1,2}$')
            if year_month_pattern.any():
                # Create a temporary column for sorting
                formatted_df['_sort_key'] = pd.to_datetime(formatted_df[x_axis], errors='coerce')
                # Sort by date
                formatted_df = formatted_df.sort_values('_sort_key')
                # Format the display values (YYYY-MM to "Mon YYYY")
                formatted_df[x_axis] = formatted_df['_sort_key'].apply(
                    lambda x: x.strftime('%b %Y') if pd.notnull(x) else x
                )
                # Drop temporary column
                formatted_df.drop('_sort_key', axis=1, inplace=True)
                
                return formatted_df
        
        # Check if the column contains month information (numeric values 1-12 or strings like '01', '02', etc.)
        is_month_column = False
        
        # Check if values are strings that could be months
        if formatted_df[x_axis].dtype == 'object':
            # Check if values match month pattern (01-12 or 1-12)
            month_pattern = formatted_df[x_axis].str.match(r'^(0?[1-9]|1[0-2])$')
            if month_pattern.any():
                is_month_column = True
                
        # Check if values are integers that could be months
        elif pd.api.types.is_numeric_dtype(formatted_df[x_axis]):
            if formatted_df[x_axis].min() >= 1 and formatted_df[x_axis].max() <= 12:
                is_month_column = True
        
        # If it's a month column, convert to month names
        if is_month_column:
            # Convert to integers first to handle both string and numeric formats
            formatted_df['_month_num'] = pd.to_numeric(formatted_df[x_axis], errors='coerce')
            # Convert to month names
            formatted_df[x_axis] = formatted_df['_month_num'].apply(
                lambda x: calendar.month_abbr[int(x)] if pd.notnull(x) and 1 <= int(x) <= 12 else x
            )
            # Drop temporary column
            formatted_df.drop('_month_num', axis=1, inplace=True)
    
    return formatted_df

def create_visualization(df, x_axis, y_axis, visual_type="bar",group_col=None):
    """
    Create the appropriate visualization based on the specified type and axes.
    
    Args:
        df: DataFrame containing the data
        x_axis: Column name to use for x-axis
        y_axis: Column name to use for y-axis
        visual_type: Type of visualization to create
        
    Returns:
        fig: Matplotlib figure object
    """
    # Limit to a reasonable number of data points
    chart_data = df.head(500) if len(df) > 500 else df
    # Format months and sort dates if applicable
    chart_data = format_month_and_sort(chart_data, x_axis)

    # Additional sorting for group columns if they contain dates
    if group_col and group_col in chart_data.columns:
        chart_data = format_month_and_sort(chart_data, group_col)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # Handle different visualization types
        if visual_type == "line":
            create_line_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "bar":
            create_bar_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "area":
            create_area_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "scatter":
            create_scatter_plot(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "histogram":
            create_histogram(chart_data, y_axis, ax)
            
        elif visual_type == "box":
            create_box_plot(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "pie":
            create_pie_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "heatmap":
            create_heatmap(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "bubble":
            create_bubble_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "grouped_bar":
            create_grouped_bar(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "clustered_bar":
            create_clustered_bar(chart_data, x_axis, y_axis, ax, group_col)
        else:
            # Default to bar chart if unsupported visualization type
            create_bar_chart(chart_data, x_axis, y_axis, ax)
        
        # Common chart formatting
        ax.set_xlabel(x_axis.replace('_', ' ').title())
        ax.set_ylabel(y_axis.replace('_', ' ').title())
        # ax.set_title(f'{visual_type.replace("_", " ").title()} Chart: {y_axis.replace("_", " ").title()} by {x_axis.replace("_", " ").title()}')
        ax.set_title(f'{visual_type.replace("_", " ").title()}: {y_axis.replace("_", " ").title()} by {x_axis.replace("_", " ").title()}')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating {visual_type} chart: {e}")
        # Create a fallback visualization
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Could not create visualization: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
# Optional: Debug function to check if sorting is working
def debug_date_sorting(df, x_axis):
    """Debug function to check if date sorting is working correctly."""
    print(f"Original order of {x_axis}:")
    print(df[x_axis].head(10).tolist())
    
    formatted_df = format_month_and_sort(df, x_axis)
    print(f"\nAfter formatting and sorting:")
    print(formatted_df[x_axis].head(10).tolist())
    
    return formatted_df


def create_line_chart(df, x_axis, y_axis, ax):
    """Create a line chart with the given x and y axes."""
    # Handle datetime x-axis
    if pd.api.types.is_datetime64_any_dtype(df[x_axis]):
        df = df.sort_values(by=x_axis)
        ax.plot(df[x_axis], df[y_axis], marker='o', linestyle='-', color='#1f77b4')
    
    # Handle numeric x-axis
    elif pd.api.types.is_numeric_dtype(df[x_axis]):
        df = df.sort_values(by=x_axis)
        ax.plot(df[x_axis], df[y_axis], marker='o', linestyle='-', color='#1f77b4')
    
    # Handle categorical x-axis
    else:
        # PRESERVE THE ORDER from the dataframe (which was sorted by format_month_and_sort)
        # Get unique values in the order they appear in the dataframe
        x_order = df[x_axis].drop_duplicates().tolist()
        
        # If too many categories, aggregate and take top N
        if df[x_axis].nunique() > 15:
            # Group by x_axis but preserve order
            grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
            # Sort by values and take top 15
            top_15_keys = grouped.sort_values(ascending=False).head(15).index.tolist()
            # Filter the original order to only include top 15
            filtered_order = [x for x in x_order if x in top_15_keys]
            # Reindex to maintain the original chronological order
            grouped = grouped.reindex(filtered_order)
            
            ax.plot(range(len(grouped)), grouped.values, marker='o', linestyle='-', color='#1f77b4')
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        else:
            # Group by x_axis but DON'T sort (sort=False preserves original order)
            grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
            
            ax.plot(range(len(grouped)), grouped.values, marker='o', linestyle='-', color='#1f77b4')
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_bar_chart(df, x_axis, y_axis, ax):
    """Create a bar chart with the given x and y axes."""
    # If too many categories, aggregate and take top N
    if not pd.api.types.is_numeric_dtype(df[x_axis]) and df[x_axis].nunique() > 15:
        # Preserve original order while getting top values
        x_order = df[x_axis].drop_duplicates().tolist()
        grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
        
        # Get top 15 by value but maintain chronological order
        top_15_keys = grouped.sort_values(ascending=False).head(15).index.tolist()
        filtered_order = [x for x in x_order if x in top_15_keys]
        grouped = grouped.reindex(filtered_order)
        
        grouped.plot(kind='bar', ax=ax, color='#2ca02c')
        
    elif pd.api.types.is_numeric_dtype(df[x_axis]) and df[x_axis].nunique() > 15:
        # For numeric x with many values, create bins
        bins = min(15, int(df[x_axis].nunique() / 5) + 1)
        df['binned'] = pd.cut(df[x_axis], bins=bins)
        grouped = df.groupby('binned', sort=False)[y_axis].mean()
        grouped.plot(kind='bar', ax=ax, color='#2ca02c')
        
    else:
        # Check if this looks like date/month data
        if (df[x_axis].dtype == 'object' and 
            any(month in str(df[x_axis].iloc[0]) for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])):
            # For date data, preserve chronological order
            grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
            grouped.plot(kind='bar', ax=ax, color='#2ca02c')
        else:
            # For non-date categorical data, sort by values for better readability
            grouped = df.groupby(x_axis, sort=False)[y_axis].mean().sort_values(ascending=False)
            grouped.plot(kind='bar', ax=ax, color='#2ca02c')
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def create_area_chart(df, x_axis, y_axis, ax):
    """Create an area chart with the given x and y axes."""
    # Handle datetime or numeric x-axis
    if pd.api.types.is_datetime64_any_dtype(df[x_axis]) or pd.api.types.is_numeric_dtype(df[x_axis]):
        df = df.sort_values(by=x_axis)
        ax.fill_between(df[x_axis], df[y_axis], alpha=0.5, color='#9467bd')
        ax.plot(df[x_axis], df[y_axis], color='#7f5e91')
    
    # Handle categorical x-axis
    else:
        # PRESERVE THE ORDER from the dataframe (which was sorted by format_month_and_sort)
        # Get unique values in the order they appear in the dataframe
        x_order = df[x_axis].drop_duplicates().tolist()
        
        # If too many categories, aggregate and take top N
        if df[x_axis].nunique() > 15:
            # Group by x_axis but preserve order
            grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
            # Sort by values and take top 15
            top_15_keys = grouped.sort_values(ascending=False).head(15).index.tolist()
            # Filter the original order to only include top 15
            filtered_order = [x for x in x_order if x in top_15_keys]
            # Reindex to maintain the original chronological order
            grouped = grouped.reindex(filtered_order)
            
            x_vals = range(len(grouped))
            ax.fill_between(x_vals, grouped.values, alpha=0.5, color='#9467bd')
            ax.plot(x_vals, grouped.values, color='#7f5e91')
            ax.set_xticks(x_vals)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        else:
            # Group by x_axis but DON'T sort (sort=False preserves original order)
            grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
            
            x_vals = range(len(grouped))
            ax.fill_between(x_vals, grouped.values, alpha=0.5, color='#9467bd')
            ax.plot(x_vals, grouped.values, color='#7f5e91')
            ax.set_xticks(x_vals)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_scatter_plot(df, x_axis, y_axis, ax):
    """Create a scatter plot with the given x and y axes."""
    # Ensure both axes are numeric
    if not pd.api.types.is_numeric_dtype(df[x_axis]):
        raise ValueError(f"X-axis column '{x_axis}' must be numeric for scatter plots")
    if not pd.api.types.is_numeric_dtype(df[y_axis]):
        raise ValueError(f"Y-axis column '{y_axis}' must be numeric for scatter plots")
    
    # Create scatter plot
    ax.scatter(df[x_axis], df[y_axis], alpha=0.6, color='#ff7f0e', edgecolor='k', linewidth=0.5)
    
    # Add best fit line
    try:
        z = np.polyfit(df[x_axis], df[y_axis], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df[x_axis].min(), df[x_axis].max(), 100)
        ax.plot(x_range, p(x_range), "r--", alpha=0.8)
    except:
        pass  # Skip trend line if there's an error
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_histogram(df, column, ax):
    """Create a histogram for the specified column."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric for histograms")
    
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color='#d62728')
    ax.set_xlabel(column.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)

def create_box_plot(df, x_axis, y_axis, ax):
    """Create a box plot with the given x and y axes."""
    if not pd.api.types.is_numeric_dtype(df[y_axis]):
        raise ValueError(f"Y-axis column '{y_axis}' must be numeric for box plots")
    
    # If too many categories on x-axis, limit to top N by average y value
    if not pd.api.types.is_numeric_dtype(df[x_axis]) and df[x_axis].nunique() > 10:
        # Preserve original order while getting top values
        x_order = df[x_axis].drop_duplicates().tolist()
        grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
        
        # Get top 10 by value but maintain chronological order
        top_10_keys = grouped.sort_values(ascending=False).head(10).index.tolist()
        filtered_order = [x for x in x_order if x in top_10_keys]
        
        # Filter dataframe to only include top categories in original order
        filtered_df = df[df[x_axis].isin(filtered_order)]
        # Create a categorical type with our custom order
        filtered_df[x_axis] = pd.Categorical(filtered_df[x_axis], categories=filtered_order, ordered=True)
        
        sns.boxplot(x=x_axis, y=y_axis, data=filtered_df, ax=ax, palette='viridis', order=filtered_order)
    else:
        # For fewer categories, preserve the original order
        x_order = df[x_axis].drop_duplicates().tolist()
        sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax, palette='viridis', order=x_order)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

def create_pie_chart(df, x_axis, y_axis, ax):
    """Create a pie chart with the given x and y axes."""
    # Aggregate data for pie chart
    grouped = df.groupby(x_axis)[y_axis].sum()
    
    # If too many categories, show top N and group others
    if grouped.shape[0] > 7:
        top = grouped.nlargest(6)
        others = pd.Series({'Others': grouped.sum() - top.sum()})
        combined = pd.concat([top, others])
        
        # Create pie chart with top 6 categories + "Others"
        combined.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, 
                         colors=plt.cm.tab10.colors, shadow=False)
    else:
        # Create pie chart with all categories
        grouped.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, 
                        colors=plt.cm.tab10.colors, shadow=False)
    
    ax.set_ylabel('')  # Remove y-axis label for pie chart

def create_heatmap(df, x_axis, y_axis, ax):
    """Create a heatmap with the given x and y axes."""
    # If both axes are categorical
    if not pd.api.types.is_numeric_dtype(df[x_axis]) and not pd.api.types.is_numeric_dtype(df[y_axis]):
        # Create pivot table with count of occurrences
        pivot_data = pd.crosstab(df[y_axis], df[x_axis])
        
        # Limit categories if too many
        if pivot_data.shape[0] > 15 or pivot_data.shape[1] > 15:
            # For columns (x-axis)
            if pivot_data.shape[1] > 15:
                top_cols = pivot_data.sum().nlargest(15).index
                pivot_data = pivot_data[top_cols]
            
            # For rows (y-axis)
            if pivot_data.shape[0] > 15:
                top_rows = pivot_data.sum(axis=1).nlargest(15).index
                pivot_data = pivot_data.loc[top_rows]
        
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='g', ax=ax)
        
    # If one axis is categorical and one is numeric
    elif not pd.api.types.is_numeric_dtype(df[x_axis]) or not pd.api.types.is_numeric_dtype(df[y_axis]):
        # Determine which is categorical and which is numeric
        cat_col = x_axis if not pd.api.types.is_numeric_dtype(df[x_axis]) else y_axis
        num_col = y_axis if cat_col == x_axis else x_axis
        
        # Create pivot table with mean of numeric column
        pivot_data = df.pivot_table(index=cat_col, values=num_col, aggfunc='mean')
        
        # Limit categories if too many
        if pivot_data.shape[0] > 15:
            pivot_data = pivot_data.nlargest(15, num_col)
            
        # Create heatmap as a 1D visualization
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.2f', ax=ax)
        
    # If both axes are numeric, create correlation heatmap
    else:
        corr = df[[x_axis, y_axis]].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)

def create_bubble_chart(df, x_axis, y_axis, ax):
    """Create a bubble chart with the given x and y axes."""
    # Ensure both axes are numeric
    if not pd.api.types.is_numeric_dtype(df[x_axis]):
        raise ValueError(f"X-axis column '{x_axis}' must be numeric for bubble charts")
    if not pd.api.types.is_numeric_dtype(df[y_axis]):
        raise ValueError(f"Y-axis column '{y_axis}' must be numeric for bubble charts")
    
    # Find a suitable column for bubble size
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                   and col != x_axis and col != y_axis]
    
    if not numeric_cols:
        # If no suitable size column, use constant size
        sizes = [100] * len(df)
    else:
        # Use the first available numeric column for bubble size
        size_col = numeric_cols[0]
        # Normalize sizes between 20 and 500
        min_size, max_size = 20, 500
        sizes = min_size + (df[size_col] - df[size_col].min()) / (df[size_col].max() - df[size_col].min()) * (max_size - min_size)
    
    # Find a suitable column for bubble color
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) 
                       and col != x_axis and col != y_axis]
    
    if not categorical_cols:
        # If no suitable color column, use a single color
        scatter = ax.scatter(df[x_axis], df[y_axis], s=sizes, alpha=0.6, 
                           c='#1f77b4', edgecolor='k', linewidth=0.5)
    else:
        # Use the first available categorical column for bubble color
        color_col = categorical_cols[0]
        
        # Preserve original order for color categories
        color_order = df[color_col].drop_duplicates().tolist()
        
        # If too many categories, limit to top N
        if len(color_order) > 7:
            # Get top 7 categories by mean of y
            grouped = df.groupby(color_col, sort=False)[y_axis].mean()
            top_7_keys = grouped.sort_values(ascending=False).head(7).index.tolist()
            # Filter original order to maintain chronology
            filtered_order = [x for x in color_order if x in top_7_keys]
            df = df[df[color_col].isin(filtered_order)]
            color_order = filtered_order
        
        # Create a categorical type with our custom order
        df[color_col] = pd.Categorical(df[color_col], categories=color_order, ordered=True)
        
        # Create a color map
        cmap = plt.cm.tab10
        norm = plt.Normalize(0, len(color_order) - 1)
        
        # Map categories to colors
        color_indices = df[color_col].cat.codes
        colors = cmap(norm(color_indices))
        
        scatter = ax.scatter(df[x_axis], df[y_axis], s=sizes, alpha=0.6, 
                           c=colors, edgecolor='k', linewidth=0.5)
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=cmap(norm(i)), markersize=10, 
                                     label=cat) for i, cat in enumerate(color_order)]
        ax.legend(handles=legend_elements, title=color_col.replace('_', ' ').title())
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_grouped_bar(df, x_axis, y_axis, ax):
    """Create a grouped bar chart with the given x and y axes."""
    # Ensure we have at least one grouping column
    grouping_cols = [col for col in df.columns if col != x_axis and col != y_axis]
    
    if not grouping_cols:
        # If no suitable grouping column is found, create a basic bar chart
        create_bar_chart(df, x_axis, y_axis, ax)
        return
    
    # Use the first available column as the grouping column
    group_col = grouping_cols[0]
    
    # Preserve original order for x-axis
    x_order = df[x_axis].drop_duplicates().tolist()
    
    # If too many categories, limit to top N
    if df[x_axis].nunique() > 10:
        # Get top 10 x values by mean of y
        grouped = df.groupby(x_axis, sort=False)[y_axis].mean()
        top_10_keys = grouped.sort_values(ascending=False).head(10).index.tolist()
        # Filter original order to maintain chronology
        filtered_order = [x for x in x_order if x in top_10_keys]
        df = df[df[x_axis].isin(filtered_order)]
        x_order = filtered_order
    
    # If too many group values, limit to top N
    if df[group_col].nunique() > 5:
        # Get top 5 group values by mean of y
        grouped = df.groupby(group_col, sort=False)[y_axis].mean()
        top_5_groups = grouped.sort_values(ascending=False).head(5).index.tolist()
        df = df[df[group_col].isin(top_5_groups)]
    
    # Create pivot table for grouped bar chart
    pivot_data = df.pivot_table(index=x_axis, columns=group_col, values=y_axis, aggfunc='mean')
    
    # Reindex to maintain original order
    pivot_data = pivot_data.reindex(x_order)
    
    # Plot grouped bar chart
    pivot_data.plot(kind='bar', ax=ax, rot=45)
    
    ax.set_xlabel(x_axis.replace('_', ' ').title())
    ax.set_ylabel(y_axis.replace('_', ' ').title())
    ax.legend(title=group_col.replace('_', ' ').title())
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

def create_clustered_bar(df, x_axis, y_axis, ax, group_col=None):
    """Create a clustered bar chart with the given x and y axes and an explicit grouping column.
    
    Args:
        df: DataFrame containing the data
        x_axis: Column name to use for x-axis
        y_axis: Column name to use for y-axis
        ax: Matplotlib axis object
        group_col: Column name to use for grouping (clustering). If None, will attempt to auto-detect.
    """
    # Helper function to check if column contains date/month data
    def is_date_like(column_data):
        if column_data.dtype != 'object':
            return False
        sample_value = str(column_data.iloc[0]) if len(column_data) > 0 else ""
        return any(month in sample_value for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # If no group column is provided, try to find a suitable one
    if group_col is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Find a suitable categorical column different from x_axis
        for col in categorical_cols:
            if col != x_axis and df[col].nunique() < 7:  # Limit to columns with few categories
                group_col = col
                break
        
        # If still no suitable column, try numeric columns that might be categorical
        if group_col is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                if col != x_axis and col != y_axis and df[col].nunique() < 7:
                    group_col = col
                    break
                    
        # Print debug information
        if group_col is not None:
            print(f"Auto-detected grouping column: {group_col} with {df[group_col].nunique()} unique values")
        else:
            print(f"Could not auto-detect a suitable grouping column. Available columns: {df.columns.tolist()}")
    
    # If no suitable grouping column is found or provided, create a colorful bar chart
    if group_col is None or group_col not in df.columns:
        print(f"No suitable grouping column found for clustered bar chart. Using regular bar chart.")
        
        # Check if x_axis contains date data to preserve order
        if is_date_like(df[x_axis]):
            # Preserve chronological order for date data
            sorted_data = df.groupby(x_axis, sort=False)[y_axis].mean()
            sorted_data = sorted_data.head(10) if len(sorted_data) > 10 else sorted_data
        else:
            # Sort by value for non-date data
            sorted_data = df.groupby(x_axis, sort=False)[y_axis].mean().sort_values(ascending=False)
            sorted_data = sorted_data.head(10) if len(sorted_data) > 10 else sorted_data
        
        # Create a color map with distinct colors for each bar
        categories = sorted_data.index
        colors = plt.cm.tab10.colors[:len(categories)]
        
        # Plot bars with different colors
        bars = ax.bar(sorted_data.index, sorted_data.values, color=colors)
        
        # Add grid and formatting
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        return  
    
    # Get original order for x_axis and group_col before filtering
    x_order = df[x_axis].drop_duplicates().tolist()
    group_order = df[group_col].drop_duplicates().tolist()
    
    # If too many categories on x-axis, limit to top N but preserve order
    if df[x_axis].nunique() > 8:
        if is_date_like(df[x_axis]):
            # For date data, take first 8 chronologically
            top_categories = x_order[:8]
        else:
            # For non-date data, take top 8 by value
            top_categories = df.groupby(x_axis, sort=False)[y_axis].mean().nlargest(8).index.tolist()
            # Preserve original order
            top_categories = [x for x in x_order if x in top_categories]
        
        filtered_df = df[df[x_axis].isin(top_categories)]
    else:
        filtered_df = df
    
    # If too many categories in group column, limit to top N but preserve order
    if df[group_col].nunique() > 6:
        if is_date_like(df[group_col]):
            # For date data, take first 6 chronologically
            top_groups = group_order[:6]
        else:
            # For non-date data, take top 6 by value
            top_groups = df.groupby(group_col, sort=False)[y_axis].mean().nlargest(6).index.tolist()
            # Preserve original order
            top_groups = [x for x in group_order if x in top_groups]
        
        filtered_df = filtered_df[filtered_df[group_col].isin(top_groups)]
    
    # Create pivot table for clustered bar chart
    # Use observed=True to avoid issues with categorical data
    pivot_data = filtered_df.pivot_table(index=x_axis, columns=group_col, values=y_axis, aggfunc='mean')
    
    # Reindex to preserve chronological order
    if is_date_like(df[x_axis]):
        # Preserve chronological order for x-axis
        available_x = [x for x in x_order if x in pivot_data.index]
        pivot_data = pivot_data.reindex(available_x)
    
    if is_date_like(df[group_col]):
        # Preserve chronological order for grouping column
        available_groups = [g for g in group_order if g in pivot_data.columns]
        pivot_data = pivot_data[available_groups]
    
    # If there's only one group, create a colorful bar chart
    if pivot_data.shape[1] == 1:
        categories = pivot_data.index
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        
        # For single group, preserve order based on data type
        if is_date_like(df[x_axis]):
            # Keep chronological order
            sorted_data = pivot_data.iloc[:, 0]
        else:
            # Sort by values for non-date data
            sorted_data = pivot_data.iloc[:, 0].sort_values(ascending=False)
        
        sorted_data = sorted_data.head(10) if len(sorted_data) > 10 else sorted_data
        
        # Plot bars with different colors
        bars = ax.bar(sorted_data.index, sorted_data.values, color=colors[:len(sorted_data)])
        
        # Add grid and formatting
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        # Plot clustered bar chart with a colorful palette
        pivot_data.plot(kind='bar', ax=ax, colormap='tab10')
        
        # Add legend and formatting
        ax.legend(title=group_col.replace('_', ' ').title())
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')