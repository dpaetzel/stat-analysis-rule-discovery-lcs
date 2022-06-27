import subprocess
import sys

import arviz as az
import seaborn as sns
import IPython
import click
import cmpbayes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_rows = 2000

plt.style.use("seaborn")

# TODO Store via PGF backend with nicer LaTeXy fonts etc.
# https://jwalton.info/Matplotlib-latex-PGF/
# matplotlib.use("pgf")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dname = "RD"
suffix = ".csv"

metrics = ["test_mean_squared_error", "elitist_complexity"]


def list_from_ls(dname):
    proc = subprocess.run(["ls", dname], capture_output=True)

    if proc.stderr != b"":
        print(proc.stderr)
        sys.exit(1)

    # Remove last element since that is an empty string always due to `ls`'s
    # final newline.
    return proc.stdout.decode().split("\n")[:-1]


def load_data():
    algorithm_names = list_from_ls(dname)
    print(algorithm_names)

    task_names = [
        n.removesuffix(suffix)
        for n in list_from_ls(f"{dname}/{algorithm_names[0]}")
        if n != f"summary{suffix}"
    ]

    dfs = []
    keys = []
    for algorithm_name in algorithm_names:
        for task_name in task_names:
            df = pd.read_csv(f"{dname}/{algorithm_name}/{task_name}{suffix}")
            dfs.append(df)
            keys.append((algorithm_name, task_name))

    df = pd.concat(dfs,
                   keys=keys,
                   names=["algorithm", "task"],
                   verify_integrity=True)

    df["test_mean_squared_error"] = -df["test_neg_mean_squared_error"]
    del df["test_neg_mean_squared_error"]

    df = df[metrics]

    assert not df.isna().any().any(), "Some values are missing"
    return df


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

    def smart_print(df):
        if latex:
            print(df.to_latex())
        else:
            print(df.to_markdown())

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

            print(f"Sample statistics for “{title}” are as follows:")
            print()
            ranks = d.apply(np.argsort, axis=1) + 1
            print(title)
            smart_print(ranks.mean())

            algorithm_labels = df.reset_index("algorithm").algorithm.unique()

            # NOTE We fix the random seed here to enable model caching.
            model = cmpbayes.Calvo(d.to_numpy(),
                                   higher_better=False,
                                   algorithm_labels=algorithm_labels).fit(
                                       num_samples=10000, random_seed=1)

            if check_mcmc:
                smart_print(az.summary(model.data_))
                az.plot_trace(model.data_)
                az.plot_rank(model.data_)

            # Join all chains, name columns.
            sample = np.concatenate(model.data_.posterior.weights)
            sample = pd.DataFrame(
                sample, columns=model.data_.posterior.algorithm_labels)
            ylabel = "algorithm"
            xlabel = "probability of having the lowest mean squared error"
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
def kruschke():
    pass
    # Kruschke: ES vs. NSLC (NSLC is basically NS but better) (rope 0.01)


if __name__ == "__main__":
    cli()
