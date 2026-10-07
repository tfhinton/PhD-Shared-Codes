import numpy as np
import pymc as pm


class GaussianDist:
    def __init__(self, label, mu, sigma):
        self.label = label
        self.mu = mu
        self.sigma = sigma

    @property
    def val(self):
        return self.mu

    @property
    def pm(self):
        return pm.Normal(self.label, mu=self.mu, sigma=self.sigma)

    def draw(self, rng):
        return float(rng.normal(self.mu, self.sigma))


class UniformDist:
    def __init__(self, label, lower, upper):
        self.label = label
        self.lower = lower
        self.upper = upper

    @property
    def val(self):
        return (self.lower+self.upper)/2

    @property
    def pm(self):
        return pm.Uniform(self.label, lower=self.lower, upper=self.upper)

    def draw(self, rng):
        return float(rng.uniform(self.lower, self.upper))


class TruncatedGaussianDist:
    '''Normal(mu, sigma) restricted to [lower, upper] (either may be None for
    unbounded on that side; the default lower=0 gives the zero-truncated normal
    used for non-negative quantities like slip).

    Unlike a Uniform, mu is free to sit outside the support: mu < 0 with
    lower=0 is a perfectly well-defined prior that piles its mass up against
    zero, which is what you want when the distribution being translated into a
    prior is consistent with no slip at all.

    ``val``/``draw`` are always strictly interior -- PyMC's interval transform
    maps the bounds to +/-inf, so a starting value exactly on a bound is an
    immediate -inf logp.
    '''

    EDGE = 1e-3     # keep val/draw this many sigma clear of a bound

    def __init__(self, label, mu, sigma, lower=0., upper=None):
        self.label = label
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)

    def _clip(self, x):
        pad = self.EDGE * self.sigma
        if self.lower is not None:
            x = max(x, self.lower + pad)
        if self.upper is not None:
            x = min(x, self.upper - pad)
        return float(x)

    @property
    def val(self):
        '''Mode of the truncated density: mu, pushed inside the support.'''
        return self._clip(self.mu)

    @property
    def pm(self):
        kw = {}
        if self.lower is not None:
            kw['lower'] = self.lower
        if self.upper is not None:
            kw['upper'] = self.upper
        return pm.TruncatedNormal(self.label, mu=self.mu, sigma=self.sigma, **kw)

    def draw(self, rng):
        from scipy.stats import truncnorm
        lo = -np.inf if self.lower is None else (self.lower - self.mu) / self.sigma
        hi = np.inf if self.upper is None else (self.upper - self.mu) / self.sigma
        x = truncnorm.rvs(lo, hi, loc=self.mu, scale=self.sigma,
                          random_state=rng)
        return self._clip(x)
