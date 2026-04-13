import arviz as az
import numpy as np
from scipy.stats import gaussian_kde

def get_maps_from_arviz(inference_data):
    
    maps = []

    prior_labels = list(inference_data.posterior.keys())
    for l in prior_labels:
        samples = inference_data.posterior[l].values
        samples = samples.flatten()
        grid, kde = az.kde(samples)
        map = grid[np.argmax(kde)]
        maps.append( map )
        
    return np.array(maps)

def get_medians_from_arviz(inference_data):
    
    medians = []

    prior_labels = list(inference_data.posterior.keys())
    for l in prior_labels:
        samples = inference_data.posterior[l].values
        samples = samples.flatten()
        median = np.median(samples)
        medians.append( median )
        
    return np.array(medians)

def plot_slip_profile(ax, patches, slips, plot_kwargs={}):
    line_xs = []
    line_ys = []
    for i, p in enumerate(patches):
        line_xs.extend([slips[i]]*2)
        line_ys.extend([p.top, p.bottom])
    ax.plot(line_xs, line_ys, **plot_kwargs)