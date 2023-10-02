import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

def trunc_norm_interval(p, mean, std, a = 0, b = 1):
    # function for defining prediction interval within a truncated normal distribution
    # p: prediction interval width (e.g. 90%)
    # mean: mean of underlying normal distribution
    # std: standard deviation of underlying normal distribution
    # a,b: boundaries

    # scale boundaries to match scipy definitions
    a, b = (a - mean) / std, (b - mean) / std
    # divide interval to quantiles
    q = p / 2
    # find mode of defined truncated normal distribution
    mode_p = truncnorm.cdf(mean, a, b, mean, std)
    # if lower quantile is out of boundaries, set lower interval at boundary
    if mode_p - q < 0:
        lb = 0
        hb = p
    # if upper quantile is out of boundaries, set upper interval at boundary
    elif mode_p + q > 1:
        hb = 1
        lb = 1 - p 
    # else, set at equal distances from mode
    else:
        lb = mode_p - q
        hb = mode_p + q
    # acquire values of quantile boundaries 
    low_val = truncnorm.ppf(lb, a, b, mean, std)
    high_val = truncnorm.ppf(hb, a, b, mean, std)

    return low_val, high_val, lb, 1-hb

def plot_trunc_norm(mean, std, a = 0, b = 1, name = None):
    # function for plotting a truncated normal distribution (prediction result)
    # mean: mean of underlying normal distribution
    # std: standard deviation of underlying normal distribution
    # a,b: boundaries
    # name: path of the file used for saving
    
    # scale boundaries to scipy definitions
    a_n, b_n = a, b
    a, b = (a - mean) / std, (b - mean) / std
    # initialise values for plotting
    x = np.linspace(0, 1, 1000)
    # generate probability distribution function based on values
    pdf = truncnorm.pdf(x, a, b, mean, std)
    
    # add some extra points for nicer visualisation
    x = list(x)
    pdf = list(pdf)
    for i in range(10):
        x.insert(0,0-i*0.01)
        x.append(1+i*0.01)
        pdf.insert(0, 0)
        pdf.append(0)
    x = np.array(x) 
    pdf = np.array(pdf)
  
    # find maximum likelihood for scaling figure
    ml = truncnorm.pdf(mean, a, b, mean, std)
    
    # generate prediction interval
    pi = trunc_norm_interval(0.90, mean, std)
    
    # construct figure
    plt.figure()
    plt.plot(x, pdf, label="Prediction\ndistribution")
    plt.fill_between(x, pdf, where = (pi[0] < x)&(pi[1] > x), alpha = 0.3)
    plt.plot([pi[0], pi[0]], [-0.05, ml*1.05], color="dimgray", linestyle="dashed",
            label="90\% Prediction\ninterval")
    plt.plot([pi[1], pi[1]], [-0.05, ml*1.05], color="dimgray", linestyle="dashed")
    plt.plot([mean, mean], [-0.05, ml*1.05], color="black", label="Point estimate")
    plt.text(mean-0.06, ml*1.13, r"$\hat{f} = %.2f$" % (mean), color = "black")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.1, ml*1.2])
    plt.xlabel("Friction factor", fontsize=14)
    plt.ylabel("Likelihood", fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(loc="best")
    plt.tight_layout()
    # save figure
    plt.savefig(name)
    
    return pi
