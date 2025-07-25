import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def create_faceted_horizontal_barchart(data, output_path, 
                                     figsize_per_facet=(8, 4)):
    """
    Create horizontal bar charts for each facet in the data.
    
    Args:
        data: Dictionary where keys are facet names and values are lists of items
        output_path: Path to save the PNG file
        figsize_per_facet: (width, height) for each subplot
    """
    
    # Calculate figure size based on number of facets
    new_data = {}
    for k, fm in data.items():
        new_fm = {str(k)[0:50]: v for k, v in list(fm.items())[0:10] if isinstance(k, (str, int, float))}
        new_data[k] = new_fm
    data = new_data
    n_facets = len(data)
    fig_width = figsize_per_facet[0]
    fig_height = figsize_per_facet[1] * n_facets
    
    # Create subplots
    fig, axes = plt.subplots(n_facets, 1, figsize=(fig_width, fig_height))
    
    # Handle case where there's only one facet
    if n_facets == 1:
        axes = [axes]
    
    # Process each facet
    for i, (facet_name, items) in enumerate(data.items()):
        ax = axes[i]
        
        # Count occurrences of each item
        counts = Counter(items)
        
        # Sort by count (descending) for better visualization
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Separate labels and values
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, alpha=0.7)
        
        # Customize the subplot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title(f'{facet_name.replace("_", " ").title()}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + 0.01 * max(values), bar.get_y() + bar.get_height()/2, 
                   str(value), ha='left', va='center', fontsize=9)
        
        # Invert y-axis so highest counts are at top
        ax.invert_yaxis()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # print(f"Faceted bar chart saved to: {output_path}")
    
