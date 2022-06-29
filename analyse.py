import subprocess
import sys
from functools import partial

import arviz as az
import click
import cmpbayes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed

pd.options.display.max_rows = 2000

plt.style.use("seaborn")

# TODO Store via PGF backend with nicer LaTeXy fonts etc.
# https://jwalton.info/Matplotlib-latex-PGF/
# matplotlib.use("pgf")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dname = "RD"
suffix = ".csv"

metrics = {
    "test_mean_squared_error": "MSE",
    "elitist_complexity": "number of rules selected"
}
tasks = {
    "combined_cycle_power_plant" : "CCPP",
    "concrete_strength" : "CS",
    "airfoil_self_noise" : "ASN",
    "energy_cool" : "EEC",
}


def list_from_ls(dname):
    proc = subprocess.run(["ls", dname], capture_output=True)

    if proc.stderr != b"":
        print(proc.stderr)
        sys.exit(1)

    # Remove last element since that is an empty string always due to `ls`'s
    # final newline.
    return proc.stdout.decode().split("\n")[:-1]


def smart_print(df, latex):
    if latex:
        print(df.to_latex())
    else:
        print(df.to_markdown())


def load_data():
    algorithm_names = list_from_ls(dname)
    print(algorithm_names)

    # task_names = [
    #     n.removesuffix(suffix)
    #     for n in list_from_ls(f"{dname}/{algorithm_names[0]}")
    #     if n != f"summary{suffix}"
    # ]
    # assert sort(task_names == tasks.keys()

    dfs = []
    keys = []
    for algorithm_name in algorithm_names:
        for task in tasks:
            df = pd.read_csv(f"{dname}/{algorithm_name}/{task}{suffix}")
            dfs.append(df)
            keys.append((algorithm_name, task))

    df = pd.concat(dfs,
                   keys=keys,
                   names=["algorithm", "task"],
                   verify_integrity=True)

    df["test_mean_squared_error"] = -df["test_neg_mean_squared_error"]
    del df["test_neg_mean_squared_error"]

    df = df[metrics.keys()]

    assert not df.isna().any().any(), "Some values are missing"
    return df


def round_to_n_sig_figs(x, n):
    decimals = -int(np.floor(np.log10(np.abs(x)))) + (n - 1)
    return x if x == 0 else np.round(x, decimals)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--latex/--no-latex",
              help="Generate LaTeX output (tables etc.)",
              default=False)
@click.option("--all-variants/--no-all-variants",
              help="Plot different variants for interpreting run/cv data",
              default=False)
@click.option(
    "--check-mcmc/--no-check-mcmc",
    help="Whether to show plots/tables for rudimentary MCMC sanity checking",
    default=False)
def calvo(latex, all_variants, check_mcmc):

    df = load_data()

    # Explore whether throwing away distributional information gives us any
    # insights.
    variants = ({
        # Mean/median over cv runs per task (n_tasks problem instances).
        "mean of":
        lambda _df: _df[metric].groupby(["algorithm", "task"]).mean().unstack(
        ).T,
        "median of":
        lambda _df: _df[metric].groupby(["algorithm", "task"]).median().
        unstack().T,
        # Each cv run as a separate problem instance (n_tasks * n_cv_runs problem
        # instances).
        "all":
        lambda _df: _df[metric].unstack(0),
    })
    # Insight: We don't want to throw away distributional information.
    if not all_variants:
        variants = {"all": variants["all"]}

    for metric in metrics:
        fig, ax = plt.subplots(len(variants))
        if not all_variants:
            ax = [ax]
        i = -1
        for mode, f in variants.items():
            d = f(df)
            i += 1
            title = f"Considering {mode} cv runs per task"

            print(
                f"Sample statistics of {metrics[metric]} for “{title}” are as follows:"
            )
            print()
            ranks = d.apply(np.argsort, axis=1) + 1
            print(title)
            smart_print(ranks.mean(), latex=latex)

            algorithm_labels = df.reset_index("algorithm").algorithm.unique()

            # NOTE We fix the random seed here to enable model caching.
            model = cmpbayes.Calvo(d.to_numpy(),
                                   higher_better=False,
                                   algorithm_labels=algorithm_labels).fit(
                                       num_samples=10000, random_seed=1)

            if check_mcmc:
                smart_print(az.summary(model.data_), latex=latex)
                az.plot_trace(model.data_)
                az.plot_rank(model.data_)

            # Join all chains, name columns.
            sample = np.concatenate(model.data_.posterior.weights)
            sample = pd.DataFrame(
                sample, columns=model.data_.posterior.algorithm_labels)
            ylabel = "algorithm"
            xlabel = f"probability of having the lowest {metrics[metric]}"
            sample = sample.unstack().reset_index(0).rename(columns={
                "level_0": ylabel,
                0: xlabel
            })

            sns.boxplot(data=sample, y=ylabel, x=xlabel, ax=ax[i])
            if all_variants:
                ax[i].set_title(title)

        plt.tight_layout()
        plt.show()


@cli.command()
@click.option("--latex/--no-latex",
              help="Generate LaTeX output (tables etc.)",
              default=False)
def ttest(latex):
    df = load_data()
    cand1 = "ES"
    cand2 = "NSLC"

    hdis = {}
    for metric in metrics:
        hdis[metrics[metric]] = {}

        print(f"# {metrics[metric]}")
        print()

        fig, ax = plt.subplots(4)
        for i, task in enumerate(tasks):

            y1 = df[metric].loc[cand1, task]
            y2 = df[metric].loc[cand2, task]
            model = cmpbayes.BayesCorrTTest(y1, y2, fraction_test=0.25).fit()

            # Compute 100(1 - alpha)% high density interval.
            alpha = 0.005
            hdi = (model.model_.ppf(alpha), model.model_.ppf(1 - alpha))
            hdis[metrics[metric]][tasks[task]] = { "lower" : hdi[0], "upper": hdi[1] }

            # Compute bounds of the plots based on ppf.
            xlower_ = model.model_.ppf(1e-6)
            xlower_ -= xlower_ * 0.07
            xupper_ = model.model_.ppf(1 - 1e-6)
            xupper_ += xupper_ * 0.07
            xlower = np.abs([xlower_, xupper_, *hdi]).max()
            xupper = -xlower

            # Compute pdf values of posterior.
            x = np.linspace(xlower, xupper, 1000)
            # y = model.model_.cdf(x)
            # x = np.arange(1e-3, 1 - 1e-3, 1e-3)
            y = model.model_.pdf(x)

            # Create DataFrame for easier seaborn'ing.
            xlabel = f"{metrics[metric]}({cand2}) - {metrics[metric]}({cand1})"
            ylabel = "density"
            data = pd.DataFrame({xlabel: x, ylabel: y})

            # Plot posterior.
            # sns.histplot(model.model_.rvs(50000),
            #              bins=100,
            #              ax=ax[i],
            #              stat="density")
            sns.lineplot(data=data, x=xlabel, y=ylabel, ax=ax[i])
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")
            ax[i].set_title(f"{tasks[task]}")

            # Add HDI lines and values.
            ax[i].vlines(x=hdi,
                         ymin=-0.1 * max(y),
                         ymax=1.1 * max(y),
                         colors="C1",
                         linestyles="dashed")
            ax[i].text(x=hdi[0],
                    y=1.15 * max(y),
                    s=round_to_n_sig_figs(hdi[0], 2),
                    ha="right",
                    va="center",
                    color="C1",
                    fontweight="bold")
            ax[i].text(x=hdi[1],
                    y=1.15 * max(y),
                    s=round_to_n_sig_figs(hdi[1], 2),
                    ha="left",
                    va="center",
                    color="C1",
                    fontweight="bold")

            print(f"## {task} ({tasks[task]})")
            print()
            print(
                f"{100 * (1 - 2 * alpha):.1f}% that difference lies in {hdi}")
            print()

        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel(xlabel)
        plt.tight_layout()
        plt.show()
        print()

    # https://stackoverflow.com/a/67575847/6936216
    hdis_ = hdis
    hdis_melt = pd.json_normalize(hdis_, sep=">>").melt()
    hdis = hdis_melt["variable"].str.split(">>", expand=True)
    hdis.columns = ["n", "metric", "task", ""]
    del hdis["n"]
    hdis["bound"] = hdis_melt["value"]
    hdis = hdis.set_index(list(hdis.columns[:-1]))
    hdis["bound"] = hdis["bound"].apply(partial(round_to_n_sig_figs, n=2))
    hdis = hdis.groupby(["metric", "task"]).agg(list)

    smart_print(hdis, latex=latex)

# TODO Consider to try tom, too, here
if __name__ == "__main__":
    cli()
