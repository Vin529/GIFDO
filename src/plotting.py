from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os

from config import LOCATION_NAME, SUBDIVISION_GENERATION
from crop import SIMULATION_CROP

CROP_NAME = SIMULATION_CROP.crop_name


def save_top_fitness_history_plot(top_fitness_history: list[float], save_directory: str) -> None:
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"save directory not found: {save_directory}")
    if not top_fitness_history:
        raise ValueError("top_fitness_history is empty")

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14
    })   

    #plot the fitness history
    plt.figure(figsize=(7, 5))
    x = np.arange(1, len(top_fitness_history) + 1)
    plt.plot(x, top_fitness_history, linewidth=2)

    #add dotted vertical line to show the generation where subdivision happens
    if len(top_fitness_history) > SUBDIVISION_GENERATION:
        plt.axvline(x=SUBDIVISION_GENERATION, color='grey', linestyle='--')

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(
        f"{CROP_NAME} – {LOCATION_NAME}\n"
        "Evolution of Best Candidate Fitness",
        multialignment='center'
    )
    plt.locator_params(axis='x', nbins=8)
    plt.grid()
    plt.tight_layout()

    plot_save_path = os.path.join(save_directory, "top_fitness_history.pdf")
    plt.savefig(plot_save_path)
    plt.close()


def save_top_half_material_use_percentage_history_plot(material_use_percentage_history: list[Counter], save_directory: str) -> None:
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"save directory not found: {save_directory}")
    if not material_use_percentage_history:
        raise ValueError("material_use_frequency_history is empty")

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14
    })    

    df = pd.DataFrame(material_use_percentage_history).fillna(0)
    #sort materials alphabetically to preserve colour assignment between runs
    df = df.reindex(sorted(df.columns), axis=1)
    df.index.name = "Generation"
    ax = df.plot(figsize=(14.32, 7), linewidth=2)

    #add dotted vertical line to show the generation where subdivision happens
    if len(material_use_percentage_history) > SUBDIVISION_GENERATION:
        ax.axvline(x=SUBDIVISION_GENERATION, color='grey', linestyle='--')

    ax.set_xlabel("Generation")
    ax.set_ylabel("Material Usage (%)")
    ax.set_title(
        f"{CROP_NAME} – {LOCATION_NAME}\n"
        "Evolution of Material Distribution in Top 50% Candidates",
        multialignment='center'
    )

    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.grid(True, linestyle="--", alpha=0.3)
    #format the legend to the right outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height])
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)

    plot_save_path = os.path.join(save_directory, "material_use_percentage_history.pdf")
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()


def save_top_half_generic_history_plot(
    variable_history: list[list[float]], 
    save_directory: str, 
    colour: str,
    y_label: str, 
    y_units: str | None,
    x_label: str = "Generation"
) -> None:
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"save directory not found: {save_directory}")    
    if not variable_history:
        raise ValueError("variable_history is empty")

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14
    })   

    variable_history_np = np.array(variable_history)
    min_values = np.min(variable_history_np, axis=1)
    max_values = np.max(variable_history_np, axis=1)
    median_values = np.median(variable_history_np, axis=1)


    plt.figure(figsize=(7, 5))
    x = np.arange(1, len(variable_history_np) + 1)
    #median line and shaded area for min to max range
    plt.plot(x, median_values, color=colour, linewidth=3, label='Median')
    plt.fill_between(x, min_values, max_values, color=colour, alpha=0.3, label='Min-Max Range')

    #add dotted vertical line to show the generation where subdivision happens
    if len(variable_history) > SUBDIVISION_GENERATION:
        plt.axvline(x=SUBDIVISION_GENERATION, color='grey', linestyle='--')

    plt.xlabel(x_label)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True)) #make sure generations are in integers
    if y_units is not None:
        y_label_text = f"{y_label} ({y_units})"
    else:
        y_label_text = y_label
    plt.ylabel(y_label_text)
    plt.title(f"{CROP_NAME} – {LOCATION_NAME}")

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path_safe_y_label = y_label.replace('/', '_').replace(' ', '_')
    plot_save_path = os.path.join(save_directory, f"top_half_{path_safe_y_label}_history.pdf")
    plt.savefig(plot_save_path)
    plt.close()
