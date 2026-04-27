import seaborn as sns
import matplotlib.pyplot as plt

def styles():
    sns.set_theme()
    sns.set_style("ticks")
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.style.use("codes.thinstyles")

def set_styles():
    sns.set_theme()
    sns.set_style("ticks")
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.style.use("codes.styles")