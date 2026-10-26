

\# M2: Adstock, Saturation and Time-Delay Analysis



\## 1. Objective



This analysis investigates advertising carry-over effects (adstock),

diminishing returns (saturation), and the time delay between advertising

spend and sales revenue across six marketing channels.



\## 2. Adstock Analysis



Geometric adstock was implemented with decay values of 0.0, 0.3, 0.5,

and 0.7. A larger decay value represents a stronger assumed carry-over

effect from previous months.



The results are saved in `outputs/tables/m2/adstock\_summary.csv`.

These decay values are scenarios for comparison and have not been

statistically fitted to the dataset.



\## 3. Saturation Analysis



Hill saturation curves were generated for all six marketing channels.

The median historical channel spend was used as the illustrative

half-saturation point, with a Hill slope of 1.5.



The curves demonstrate how a nonlinear response can represent

diminishing returns. The parameters are illustrative assumptions;

they are not estimates of actual channel effectiveness.



Figure: `outputs/figures/m2/saturation\_curves.png`



\## 4. Time-Delay Analysis



Pearson correlations were calculated between each channel's monthly

spend and sales revenue at lags of 0, 1, 2, and 3 months.



Outdoor advertising had the strongest same-month correlation

(r = 0.436). Google Search (r = 0.382), YouTube (r = 0.365),

Facebook/Instagram (r = 0.333), and TV (r = 0.325) also had positive

same-month correlations.



Radio had a low same-month correlation (r = 0.043), but its strongest

observed relationship was at a two-month lag (r = 0.354).



These findings are descriptive. The analysis does not establish that

advertising caused changes in revenue.



Figure: `outputs/figures/m2/time\_delay\_correlations.png`



Detailed results:

\- `outputs/tables/m2/time\_delay\_correlations.csv`

\- `outputs/tables/m2/time\_delay\_correlation\_matrix.csv`



\## 5. Limitations



The dataset contains only 36 monthly observations. Correlations may

reflect common trends, seasonality, pricing, promotions, or other

confounding factors. Multiple channel-and-lag comparisons also increase

the chance of observing apparently strong relationships by chance.



The saturation parameters and adstock decay values are illustrative

rather than fitted. Further work should use time-ordered validation,

appropriate controls, and model-based parameter estimation.



\## 6. Conclusion



The exploratory analysis identifies potential differences in the

timing of channel-spend relationships with revenue. Outdoor has the

largest same-month correlation, while Radio's largest observed

correlation occurs at a two-month lag. These patterns can inform

subsequent modelling but should not be treated as causal estimates or

confirmed budget-allocation recommendations.



