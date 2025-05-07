import os

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use(["science", "ieee"])


def create_df_hashmap():
    # crawl the results directory and create a dictionary of dataframes
    # by reading the CSV files
    cache_names_to_df = {}
    for filename in os.listdir("results"):
        if filename.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join("results", filename))
            # Extract the cache name from the filename
            cache_name = filename.split("Simulation")[0]
            # Store the DataFrame in the dictionary
            cache_names_to_df[cache_name] = df
    return cache_names_to_df


def generate_all_plots():
    cache_names_to_df = create_df_hashmap()

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
        plt.savefig(f"results/{attr}_plot.eps", format="eps")
        plt.savefig(f"results/{attr}_plot.png", format="png")

    generate_plot_for_attr(attr="hit_rate")
    generate_plot_for_attr(attr="miss_rate")


if __name__ == "__main__":
    generate_all_plots()
    print("Plots generated and saved in the results directory.")
