# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# %% [markdown]
# *Import data*

# %%
tickers = ['TSLA','GOOGL', 'AMZN','PLTR', 'SHOP', 'SNOW', 'NVDA']

for ticker in tickers:
    data = yf.download(tickers, start="2020-10-01", end="2022-01-01")
    data = data['Close']
    simple_returns = data.pct_change()
    simple_returns = simple_returns.dropna()
    simple_returns.head()

# %%
mean_returns = simple_returns.mean()
print(mean_returns)
covariance_matrix = simple_returns.cov()
print(covariance_matrix)

# %%
weights = np.random.random(len(simple_returns.columns))
weights /= np.sum(weights)
print(weights)

# %%
simple_returns['portfolio'] = simple_returns.dot(weights)
print(simple_returns)

# %% [markdown]
# # Historical VaR and CVaR

# %% [markdown]
# The historical simulation method assumes that the past performance of a portfolio is a good indicator of its performance in the near future. This method reorganizes actual historical returns by ranking them from the worst to the best. It assumes the recurrence of the trend, from a risk perspective

# %% [markdown]
# Unlike the other two methods, the historical simulation method does not need any distributional assumption to estimate VaR.

# %%
# Calculate historical VaR(95)
portfolio_returns = simple_returns['portfolio']
var_95 = np.percentile(portfolio_returns, 5)
print('The historical VaR is:', var_95, "%")
cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
print('The historical CVaR is:', cvar_95, "%")
cvar_95_min = portfolio_returns[portfolio_returns <= var_95].min()
print('Max historical loss:', cvar_95_min, "%")

# Sort the returns for plotting
sorted_rets = portfolio_returns.sort_values(ascending=True)

InitialInvestment = 10000
print('Value at Risk 95th CI in $    :      ', round(InitialInvestment*var_95,2), "$")
print('Conditional VaR 95th CI in $  :      ', round(InitialInvestment*cvar_95,2), "$")

# %%
plt.figure(figsize=(8,6))
plt.hist(sorted_rets, bins=25)
plt.title('Daily Returns Frequency')
plt.xlabel('Daily Returns (Percent)')
plt.ylabel('Frequency')
plt.axvline(x=var_95, color='green', label='VaR 95: %0.4f' % var_95)  # plot the historical VaR on the histogram
plt.axvline(x=cvar_95, color='red', label='CVaR 95: %0.4f' % cvar_95)  # plot the historical CVar on the histogram
plt.legend()
plt.show()

# %%
# Creating a column where the portfolio returns were below the VaR_95
simple_returns["violations"] = portfolio_returns[portfolio_returns <= var_95]
violations = simple_returns["violations"]
violations[violations <= 0] = 1
violations = violations.replace(np.nan, 0)
print(violations.value_counts())

# %% [markdown]
# Kupiec Test (1995): verify if the number of violations is consistent with the violations predicted by the model;

# %%
import vartests

vartests.kupiec_test(violations, var_conf_level=0.95, conf_level=0.95)

# %% [markdown]
# The results suggest that the var model introduced allows to identify the falilure rate.

# %% [markdown]
# Christoffersen and Pelletier Test (2004): also known as Duration Test. Duration is time between violations of VaR. It tests if VaR Model has quickly response to market movements by consequence the violations do not form volatility clusters. This test verifies if violations has no memory i.e. should be independent.

# %%
vartests.duration_test(violations, conf_level=0.95)

# %%
vartests.failure_rate(violations)

# %% [markdown]
# Based on the failure rate, the historical VaR is adquate. In fact, it corresponds to the 5% used with the CL=95%.

# %% [markdown]
# *Historical VaR and CVaR for the 100 recent days*

# %%
tail_returns = portfolio_returns.tail(100)
print(tail_returns)
var95_100 = np.percentile(tail_returns, 5)
print('The var for the 100 recent days is: %0.4f' % var95_100, "%")
cvar95_100 = tail_returns[tail_returns <= var95_100].mean()
print('The cvar for the 100 recent days is: %0.4f' % cvar95_100, "%")
cvar_95_min = portfolio_returns[portfolio_returns <= var_95].min()
print('Max historical 100 day loss:', cvar_95_min, "%")

InitialInvestment = 10000
print('Value at Risk 95th CI in $   :      ', round(InitialInvestment*var95_100,2), "$")
print('Conditional VaR 95th CI in $ :      ', round(InitialInvestment*cvar95_100,2), "$")

# %%
plt.figure(figsize=(8,6))
plt.hist(tail_returns, bins=25)
plt.title('Daily Returns Frequency Over 100 Days')
plt.xlabel('Daily Returns (Percent)')
plt.ylabel('Frequency')
plt.axvline(x=var95_100, color='green', label='VaR 95: %0.4f' % var95_100)  # plot the historical VaR on the histogram
plt.axvline(x=cvar95_100, color='red', label='CVaR 95: %0.4f' % cvar95_100)  # plot the historical CVar on the histogram
plt.legend()
plt.show()

# %% [markdown]
# # Parametric VaR and CVaR

# %% [markdown]
# The parametric method is also called the variance-covariance method. This method looks at the price changes of an investment over a lookback period and computes a portfolio’s maximum loss using probability theory. It uses the standard deviation and the mean of the price returns of an asset (in this case of a portfolio) as the parameters. The maximum loss within a specific confidence level is calculated, assuming asset price returns and volatility follow a normal distribution.

# %%
from scipy.stats import norm, t

# %%
portfolio_mean_return = portfolio_returns.mean()
print(portfolio_mean_return)
portfolio_std = portfolio_returns.std()
print(portfolio_std)

# %%
def var_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        VaR = norm.ppf(1-alpha/100)*portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        x = dof
        VaR = np.sqrt((x-2)/x) * t.ppf(1-alpha/100, x) * portfolioStd - portofolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return VaR

# %%
def cvar_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        CVaR = (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))* portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        y = dof
        z = t.ppf(alpha/100, y)
        CVaR = -1/(alpha/100) * (1-y)**(-1) * (y-2+z**2) * t.pdf(z, y) * portfolioStd - portfolio_mean_return
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return CVaR

# %%
normal_VaR = var_parametric(portfolio_mean_return, portfolio_std, distribution='normal')
normal_CVaR = cvar_parametric(portfolio_mean_return, portfolio_std, distribution='normal')

normal_VaR = -normal_VaR

tVaR = var_parametric(portfolio_mean_return, portfolio_std, distribution='t-distribution')
tCVaR = cvar_parametric(portfolio_mean_return, portfolio_std, distribution='t-distribution')

tVaR = -tVaR

print("Normal VaR 95th CI in $       :      ", round(InitialInvestment*normal_VaR,2), "$")
print("Normal CVaR 95th CI in $      :      ", round(InitialInvestment*normal_CVaR,2), "$")
print("t-dist VaR 95th CI in $       :      ", round(InitialInvestment*tVaR,2), "$")
print("t-dist CVaR 95th CI in $      :      ", round(InitialInvestment*tCVaR,2), "$")

# %%
simple_returns['normal_violations'] = simple_returns['portfolio'] <= normal_VaR
simple_returns['normal_violations']= simple_returns['normal_violations'].astype(float)
simple_returns['normal_violations'].replace('False', 0)
simple_returns['normal_violations'].replace('True', 1)
violations_normal= simple_returns['normal_violations']
print(violations_normal.value_counts())
print(violations_normal.unique())
print(violations_normal.describe())

# %%
vartests.kupiec_test(violations_normal, var_conf_level=0.95, conf_level=0.95)

# %%
vartests.failure_rate(violations_normal)

# %%
simple_returns['t_violations'] = simple_returns['portfolio'] <= tVaR
simple_returns['t_violations']= simple_returns['t_violations'].astype(float)
simple_returns['t_violations'].replace('False', 0)
simple_returns['t_violations'].replace('True', 1)
violations_t= simple_returns['t_violations']
print(violations_t.value_counts())
print(violations_t.unique())
print(violations_t.describe())

# %%
vartests.kupiec_test(violations_t, var_conf_level=0.95, conf_level=0.95)

# %%
vartests.failure_rate(violations_t)

# %% [markdown]
# # Monte carlo daily VaR

# %% [markdown]
# Monte Carlo simulation is a method that randomly generates trials without providing any information about the underlying methodology. This method of VaR computation is somewhat similar to the historical simulation method. However, Monte Carlo simulation generates random numbers to estimate the return of an asset. It neither uses historical data of returns nor assumes a recurrence.

# %%
number_simulations = 2500
T = 100 # forecast period 
S0 = portfolio_returns.iloc[-1] # Most recent return

# Aggregate the returns
sim_returns = []

# Loop through 2500 simulations
for i in range(number_simulations):

    # Generate the Random Walk
    rand_rets = np.random.normal(portfolio_mean_return, portfolio_std, T) 
    sim_returns.append(rand_rets)
    cumulative_returns = rand_rets + 1
    forecast_returns = S0 * (cumulative_returns.cumprod())

    plt.plot(forecast_returns)
    plt.title('Monte Carlo Simulation - Portfolio returns over time')
    plt.xlabel('Time (Days)')
    plt.ylabel('Portfolio Returns ($)')
plt.show()

# Calculate the VaR(95)
mvar_95 = np.percentile(sim_returns, 5)
print("The monte carlo parametric VaR(95): ", round(100*mvar_95, 2),"%")
print("The monte carlo parametric VaR(95):", round(InitialInvestment*mvar_95,2),"$")

# %% [markdown]
# # Monte Carlo in the end of 100 days

# %%
mc_sims = 2500 # number of simulations
T = 100 #timeframe in days
meanM = np.full(shape=(T, len(weights)), fill_value=portfolio_mean_return)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000
for m in range(0, mc_sims):
    # MC loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covariance_matrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*InitialInvestment

# %%
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value $')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show

# %%
# Using only the last value of the forecast for calculating VaR
portResults = pd.Series(portfolio_sims[-1,:])
print(portResults.head())
VaR_MC = np.percentile(portResults, 5)
print(VaR_MC)
belowVaR = portResults[portResults <= VaR_MC]
print(belowVaR.head())
CVaR_MC = sum(belowVaR)/len(belowVaR)
print(CVaR_MC)
mVaR = initialPortfolio - VaR_MC
mCVaR = initialPortfolio - CVaR_MC
print('VaR ${}'.format(round(mVaR,2)))
print('CVaR ${}'.format(round(mCVaR,2)))

# %%
portResults = pd.Series(portfolio_sims[-1,:])
df_mc = portResults.to_frame(name = "portResults")
print(df_mc)

# %%
print(belowVaR)

# %% [markdown]
# Backtesting Monte Carlo VaR 

# %%
df_mc["violations_mc"] = df_mc['portResults'] <= VaR_MC
df_mc["violations_mc"]= df_mc["violations_mc"].astype(float)
df_mc["violations_mc"].replace('False', 0)
df_mc["violations_mc"].replace('True', 1)
mc_violations = df_mc["violations_mc"]
print(mc_violations.value_counts())
print(mc_violations.describe())

# %%
vartests.kupiec_test(mc_violations, var_conf_level=0.95, conf_level=0.95)

# %%
vartests.duration_test(mc_violations, conf_level=0.95)

# %%
vartests.failure_rate(violations)

# %% [markdown]
# *Comparising between VaR´s and CVaR´s*

# %%
# Historical
print('Historical VaR 95th CI ${}'.format(round(InitialInvestment*var_95,2)))
print('Historical CVaR 95th CI ${}'.format(round(InitialInvestment*cvar_95,2)))

# 100 most recent returns historical
print('Historical 100 most recent returns VaR 95th CI ${}'.format(round(InitialInvestment*var95_100,2)))
print('Historical 100 most recent returns CVaR 95th CI ${}'.format(round(InitialInvestment*cvar95_100,2)))

# Parametric
print('Normal VaR 95th CI in ${}'.format(round(InitialInvestment*normal_VaR,2)))
print('Normal CVaR 95th CI in ${}'.format(round(InitialInvestment*-normal_CVaR,2)))
print('t-dist VaR 95th CI in ${}'.format(round(InitialInvestment*tVaR,2)))
print('t-dist CVaR 95th CI in ${}'.format(round(InitialInvestment*-tCVaR,2)))

# Monte carlo for 1 day period
print("Monte carlo VaR for 1 day period ${}".format(round(InitialInvestment*mvar_95,2),"$"))

# Monte carlo for 100 day period
print('Monte carlo VaR for 100 day period ${}'.format(round(-mVaR,2)))
print('Monte carlo CVaR for 100 day period ${}'.format(round(-mCVaR,2)))

# %% [markdown]
# # Conclusion

# %% [markdown]
# * Using the historical method, we expect in the worse 5% percentile to lose 346.18$. And beyond that level, we excepect to lose 462.71$.
# 
# * We can see that both parametric and monte carlo methods provide lower VaR than the historical method.
# 
# * All the methods through backtesting provide adquate results.
# 
# * Using the monte carlo method at the end of 100 days, we expect in the worse 5% percentile to lose 1277.16$. And beyond that level, we excepect to lose 1991.86$.


