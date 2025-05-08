import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use(["science", "ieee"])


def create_df_hashmap(results_path):
    # Crawl the results directory and create a dictionary of dataframes
    # by reading the CSV files
    cache_names_to_df = {}
    for filename in os.listdir(results_path):
        if filename.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(results_path, filename))
            # Extract the cache name from the filename
            cache_name = filename.split("Simulation")[0]
            # Store the DataFrame in the dictionary
            cache_names_to_df[cache_name] = df
    return cache_names_to_df


def generate_all_plots(results_path):
    cache_names_to_df = create_df_hashmap(results_path)

    # Plot all hit rates in the same subplot
    def generate_plot_for_attr(attr: str = "hit_rate"):
        plt.figure(figsize=(10, 6))
        for key, df in cache_names_to_df.items():
            # plot only every 10 values
            plt.plot(df["requests"], df[attr], label=f"{key}")

        plt.title(f"{attr} for different eviction policies")
        plt.xlabel("Requests")
        plt.ylabel(attr)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Save the plot as a PNG and EPS file
        plt.savefig(os.path.join(results_path, f"{attr}_plot.eps"), format="eps")
        plt.savefig(os.path.join(results_path, f"{attr}_plot.png"), format="png")

    generate_plot_for_attr(attr="hit_rate")
    generate_plot_for_attr(attr="miss_rate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from simulation results."
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to the directory containing the simulation result CSV files.",
    )
    args = parser.parse_args()

    generate_all_plots(args.results_path)
    print(f"Plots generated and saved in the '{args.results_path}' directory.")
