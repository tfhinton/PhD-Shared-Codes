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