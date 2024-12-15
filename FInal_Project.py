import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from math import sin, pi, exp, log
from sklearn.preprocessing import MinMaxScaler
from deap import base, creator, tools, algorithms

# Function to read and clean CSV data
def read_and_clean_csv(file_path, delimiter="\t"):
    try:
        df = pd.read_csv(file_path, encoding="utf-16", delimiter=delimiter)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1", delimiter=delimiter)
    
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: x.strip("ÿþ").strip())
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: float(str(x).replace("$", "").replace(",", "").replace("%", "")) 
            if isinstance(x, str) and any(c in x for c in ["$", ",", "%"]) 
            else x
        )
    return df

# Load and normalize datasets
def load_and_normalize_data():
    sale_price = read_and_clean_csv("median sale price.csv")[["Median Sale Price"]].dropna()
    days_on_market = read_and_clean_csv("median days on market.csv")[["median_days_on_market"]].dropna()
    homes_sold = read_and_clean_csv("homes sold.csv")[["adjusted_average_homes_sold"]].dropna()
    new_list_ppsf = read_and_clean_csv("median new list ppsf.csv")[["Median New Listing Ppsf"]].dropna()
    
    scaler = MinMaxScaler()
    sale_price = pd.DataFrame(scaler.fit_transform(sale_price), columns=["Median Sale Price"])
    days_on_market = pd.DataFrame(scaler.fit_transform(days_on_market), columns=["median_days_on_market"])
    homes_sold = pd.DataFrame(scaler.fit_transform(homes_sold), columns=["adjusted_average_homes_sold"])
    new_list_ppsf = pd.DataFrame(scaler.fit_transform(new_list_ppsf), columns=["Median New Listing Ppsf"])
    
    return sale_price, days_on_market, homes_sold, new_list_ppsf

# Fitness function for financial portfolio optimization
def evaluate_portfolio(individual, sale_price, days_on_market, homes_sold, new_list_ppsf):
    sale_price = sale_price.values.flatten().astype(float)
    days_on_market = days_on_market.values.flatten().astype(float)
    homes_sold = homes_sold.values.flatten().astype(float)
    new_list_ppsf = new_list_ppsf.values.flatten().astype(float)
    individual = np.array(individual)

    min_length = min(len(sale_price), len(individual))
    sale_price = sale_price[:min_length]
    days_on_market = days_on_market[:min_length]
    homes_sold = homes_sold[:min_length]
    new_list_ppsf = new_list_ppsf[:min_length]
    individual = individual[:min_length]

    total_price = np.sum(sale_price * individual)
    total_days = np.sum(days_on_market * individual)
    total_sold = np.sum(homes_sold * individual)
    total_ppsf = np.sum(new_list_ppsf * individual)

    fitness = total_sold - (0.5 * total_days) - (0.3 * total_ppsf) - (0.2 * total_price)
    return fitness,

# Benchmark functions
def m1(x):
    return sin(5 * pi * x) ** 6

def m4(x):
    if x < 0 or x > 1:
        return 0
    try:
        term1 = exp(-2 * log(2) * ((x - 0.08) / 0.854) ** 2)
        term2 = sin(5 * pi * (x ** 0.75 - 0.05)) ** 6
        return term1 * term2
    except ValueError:
        return 0

def evaluate_benchmark(individual, benchmark_function):
    fitness = sum(benchmark_function(x) for x in individual)
    return fitness,

# Niching techniques
def apply_crowding_distance(population):
    tools.emo.assignCrowdingDist(population)

def apply_sharing_function(population, sigma_share=0.1, alpha=2):
    for ind in population:
        shared_fitness = ind.fitness.values[0]
        for other in population:
            if ind != other:
                distance = np.linalg.norm(np.array(ind) - np.array(other))
                if distance < sigma_share:
                    shared_fitness -= (1 - (distance / sigma_share) ** alpha)
        ind.fitness.values = (shared_fitness,)

# GA Setup
def setup_toolbox(evaluate_function, **kwargs):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 100)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_function, **kwargs)

    return toolbox

def run_ga(toolbox, ngen=50, niching_method=None):
    population = toolbox.population(n=100)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    logbook = tools.Logbook()
    logbook.header = ["gen", "max", "avg", "min"]

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population[:] = toolbox.select(offspring, len(population))

        if niching_method == "crowding":
            apply_crowding_distance(population)
        elif niching_method == "sharing":
            apply_sharing_function(population)

        record = stats.compile(population)
        logbook.record(gen=gen, **record)

    hall_of_fame.update(population)
    return population, hall_of_fame, logbook

# Function to save optimization results to a CSV file
def save_results_to_csv(hall_of_fame, sale_price, days_on_market, homes_sold, new_list_ppsf):
    """
    Saves the best individual's results and key metric contributions to a CSV file.
    """
    best_individual = np.array(hall_of_fame[0], dtype=float)

    # Determine the minimum length across all metrics
    min_length = min(
        len(sale_price), len(days_on_market), len(homes_sold), len(new_list_ppsf), len(best_individual)
    )

    # Truncate all arrays to the minimum length
    sale_price = sale_price.values.flatten()[:min_length]
    days_on_market = days_on_market.values.flatten()[:min_length]
    homes_sold = homes_sold.values.flatten()[:min_length]
    new_list_ppsf = new_list_ppsf.values.flatten()[:min_length]
    best_individual = best_individual[:min_length]

    # Calculate contributions
    contributions = [
        np.sum(sale_price * best_individual),
        np.sum(days_on_market * best_individual),
        np.sum(homes_sold * best_individual),
        np.sum(new_list_ppsf * best_individual),
    ]

    # Define metrics and truncate weights
    metrics = ["Sale Price", "Days on Market", "Homes Sold", "New List PPSF"]
    weights = best_individual[:len(metrics)]  # Match number of metrics

    # Create the data dictionary for the DataFrame
    data = {
        "Metric": metrics,
        "Contribution": contributions,
        "Weights": weights.tolist()
    }

    # Create and save the DataFrame
    df = pd.DataFrame(data)
    df.to_csv("optimization_results.csv", index=False)
    print("Results saved to optimization_results.csv")



# Visualization for fitness evolution
def visualize_fitness(logbook, title):
    generations = logbook.select("gen")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")
    min_fitness = logbook.select("min")

    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness, label="Max Fitness", color="green")
    plt.plot(generations, avg_fitness, label="Avg Fitness", color="blue")
    plt.plot(generations, min_fitness, label="Min Fitness", color="red")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Radar chart visualization
def detailed_portfolio_visualization_radar(hall_of_fame, sale_price, days_on_market, homes_sold, new_list_ppsf):
    portfolio_weights = np.array(hall_of_fame[0], dtype=float)
    min_length = min(len(sale_price), len(portfolio_weights))
    sale_price = sale_price.values.flatten()[:min_length]
    days_on_market = days_on_market.values.flatten()[:min_length]
    homes_sold = homes_sold.values.flatten()[:min_length]
    new_list_ppsf = new_list_ppsf.values.flatten()[:min_length]
    portfolio_weights = portfolio_weights[:min_length]

    contributions = [
        np.sum(sale_price * portfolio_weights),
        np.sum(days_on_market * portfolio_weights),
        np.sum(homes_sold * portfolio_weights),
        np.sum(new_list_ppsf * portfolio_weights),
    ]
    labels = ["Sale Price", "Days on Market", "Homes Sold", "New List PPSF"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    contributions += contributions[:1]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, contributions, color="skyblue", alpha=0.4)
    ax.plot(angles, contributions, color="blue", linewidth=2)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Radar Chart of Portfolio Optimization Contributions", size=15, pad=20)
    plt.tight_layout()
    plt.show()

# Run benchmarks multiple times
def run_benchmarks(niching_method):
    for benchmark_function, name in [(m1, "M1(x)"), (m4, "M4(x)")]:
        for run in range(10):
            toolbox = setup_toolbox(evaluate_benchmark, benchmark_function=benchmark_function)
            _, hof, logbook = run_ga(toolbox, niching_method=niching_method)
            visualize_fitness(logbook, f"{name} - {niching_method.capitalize()} - Run {run + 1}")

# Main function
def main():
    # Load and normalize data for portfolio optimization
    sale_price, days_on_market, homes_sold, new_list_ppsf = load_and_normalize_data()

    # Crowding Distance Niching for Portfolio Optimization
    toolbox = setup_toolbox(
        evaluate_portfolio,
        sale_price=sale_price,
        days_on_market=days_on_market,
        homes_sold=homes_sold,
        new_list_ppsf=new_list_ppsf
    )
    _, hof_crowding, logbook_crowding = run_ga(toolbox, niching_method="crowding")
    visualize_fitness(logbook_crowding, "Portfolio Optimization (Crowding Distance)")
    detailed_portfolio_visualization_radar(hof_crowding, sale_price, days_on_market, homes_sold, new_list_ppsf)
    save_results_to_csv(hof_crowding, sale_price, days_on_market, homes_sold, new_list_ppsf)

    # Sharing Function Niching for Portfolio Optimization
    _, hof_sharing, logbook_sharing = run_ga(toolbox, niching_method="sharing")
    visualize_fitness(logbook_sharing, "Portfolio Optimization (Sharing Function)")
    detailed_portfolio_visualization_radar(hof_sharing, sale_price, days_on_market, homes_sold, new_list_ppsf)
    save_results_to_csv(hof_sharing, sale_price, days_on_market, homes_sold, new_list_ppsf)

    # Run M1 and M4 Benchmarks with Niching
    print("Running benchmarks with Crowding Distance...")
    run_benchmarks("crowding")
    print("Running benchmarks with Sharing Function...")
    run_benchmarks("sharing")

if __name__ == "__main__":
    main()
