import streamlit as st
import scipy.stats as sts
import numpy as np
import tools
import pandas as pd
from matplotlib import pyplot as plt

st.set_page_config(layout="wide")


st.markdown("# Long-term error rate calculator")

with st.expander("Intro"):
    c1, c2 = st.columns(2, gap="large")

c1.markdown("""
## What is this about?

This Streamlit app helps estimating long-term experiment rates for simple A/B tests with a binary outcome metric:

 * **Detection rate:** % of experiments in which statistically significant result will be detected;
 * **Win rate:** % of experiments where a positive statistically significant effect will be detected;
 * **Sign error rate:** % of experiments that we will detect a significant effect, but it will be of opposite sign than the true underlying effect;
 * **Exaggeration rate:** Average difference between the observed effect size that was declared as statistically significant and the true underlying effect size (an estimate of winner's curse phenomenon)

Getting to such estimates requires two inputs:

 * Sample size and baseline metric
 * Assumptions about effect size distribution

The calculator also estimates the "classical" frequentist statistical power (detection rate, but for a single experiment if repeated many times) as well as two other quantities proposed by Gelman and Carlin:

 * Type S error rate (sign error rate, but for a single experiment)
 * Type M rate (exaggeration rate, expressed as a ratio between observed and true effect size)


""")

c2.markdown("""

## Why is this useful?

 * Get estimates of long-term win and error rates for your experimentation program! Frequentist statistics don't provide these alone.
 * Explore the relationship between sample size and expected detection rates. You'll notice that MDE (and thus statistical power) doesn't affect them! That's because MDE is just an assumption, and the only thing that actually matters is sample size itself and the assumption on effect sizes. It may convince you to rethink the MDE used in your power calculations.
 * Rethink how much to focus on Type I error rates, given that null hypothesis is unlikely to be true in most settings, and thus Type S error rate may be much more useful.

 #
 #
 #
 #
 #
 #
 #
 #
 

 More details @ my blog post [Statistical guarantees and long-term error rates in A/B testing](https://aurimas.eu).

 -- Aurimas Racas, 2024
""")


power, prior, lt_rates = st.columns((1, 2, 0.5))


with power.expander("Basic inputs", expanded=True):
    sample_size = st.slider(
        "Sample size (per arm)",
        min_value=1000,
        max_value=50_000,
        step=1000,
        value=35_000,
    )

    baseline_rate = st.slider(
        "Baseline conversion rate",
        min_value=0.01,
        max_value=0.99,
        step=0.01,
        value=0.5,
    )

    MDE = st.slider(
        "Minimum detectable effect",
        min_value=0.001,
        max_value=0.1,
        step=0.001,
        value=0.01,
        format="%f",
    )

    rates = tools.calculate_rates(
        delta=MDE, sd=np.sqrt(baseline_rate * (1 - baseline_rate)), n=sample_size
    )

    cols = st.columns(3)

    cols[0].metric("Statistical power", value=f"{rates['power']:.0%}")
    cols[1].metric("Type S error rate (at MDE)", value=f"{rates['typeS']:.2%}")
    cols[2].metric("Type M ratio (at MDE)", value=f"{rates['typeM']:.2f}")

with prior.expander(
    "Assumed distribution of effect sizes across experiments", expanded=True
):
    cols = st.columns((1, 2))

    loc = cols[0].slider(
        "Location parameter (changes center point of distribution)",
        value=-0.015,
        min_value=-0.05,
        max_value=0.05,
        step=0.001,
        format="%f",
    )
    scale = cols[0].slider(
        "Scale parameter (changes how varied effect sizes are)",
        value=0.01,
        min_value=0.001,
        max_value=0.03,
        step=0.001,
        format="%f",
    )

    a = cols[0].slider(
        "$\\alpha$ parameter (changes how 'protected' the downside is)",
        value=2.0,
        min_value=1.0,
        max_value=7.0,
        step=0.1,
        format="%f",
    )

    rv = sts.gamma(loc=loc, scale=scale, a=a)

    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 1000)
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    density = rv.pdf(x)
    ax.plot(x, density, lw=1, alpha=0.6, label="gamma pdf", color="#1b9e77")
    plt.fill_between(x, rv.pdf(x), lw=1, alpha=0.6, label="gamma pdf", color="#1b9e77")
    plt.gcf().set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("white")
    plt.gca().spines["bottom"].set_color("white")
    ax.set_ylim(0, density.max() * 1.25)

    # Remove ticks on the top and right sides
    plt.tick_params(top=False, right=False)
    plt.xlabel("Effect size", fontsize=7, color="white")
    plt.ylabel("Density", fontsize=7, color="white")

    # Set x and y ticks font size
    plt.xticks(fontsize=4, color="white")
    plt.yticks(fontsize=4, color="white")

    ax.axvline(x=MDE, color="#7570b3", linestyle="--", lw=0.7)
    plt.text(
        MDE + 0.001,
        density.max() * 1.1,
        "MDE",
        fontsize=5,
        color="#7570b3",
    )

    ax.axvline(x=rv.mean(), color="#d95f02", linestyle="-.", lw=0.7)
    plt.text(
        rv.mean() + 0.001,
        density.max() * 1.2,
        "Avg. effect size",
        fontsize=5,
        color="#d95f02",
    )
    ax.set_xlim(xmin=-0.05, xmax=0.2)

    cols[1].pyplot(fig, use_container_width=True)

    summary_stats = pd.DataFrame(
        [
            {"statistic": "mean", "value": rv.mean()},
            {"statistic": "median", "value": rv.ppf(0.5)},
            {"statistic": "Q10", "value": rv.ppf(0.1)},
            {"statistic": "Q90", "value": rv.ppf(0.9)},
        ]
    )

    cols[1].write(f"Average effect size: `{rv.mean():.3f}`")
    cols[1].write(
        f"Worst 10% of experiments will have effect size below: `{rv.ppf(0.1):.3f}`"
    )
    cols[1].write(
        f"Top 10% of experiments will have effect size above: `{rv.ppf(0.9):.3f}`"
    )
    cols[1].write(f"Share of experiments that have negative impact: `{rv.cdf(0):.0%}`")

    with lt_rates.expander(label="Long term rates", expanded=True):
        st.metric(
            label="Detection rate",
            value="{:.1%}".format(
                tools.detection_rate(
                    baseline=baseline_rate,
                    required_sample_size=sample_size,
                    density_func=rv,
                )
            ),
        )

        st.metric(
            label="Detectable win rate",
            value="{:.1%}".format(
                tools.detectable_win_rate(
                    baseline=baseline_rate,
                    required_sample_size=sample_size,
                    density_func=rv,
                )
            ),
        )

        st.metric(
            label="Sign error rate",
            value="{:.1%}".format(
                tools.sign_error_rate(
                    baseline=baseline_rate,
                    required_sample_size=sample_size,
                    density_func=rv,
                )
            ),
        )

        st.metric(
            label="Avg. exaggeration rate",
            value="{:.4f}".format(
                tools.exaggeration_rate(
                    baseline=baseline_rate,
                    required_sample_size=sample_size,
                    density_func=rv,
                )
            ),
        )
