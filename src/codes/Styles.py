import seaborn as sns
import matplotlib.pyplot as plt

def styles():
    sns.set_theme()
    sns.set_style("ticks")
    plt.rcParams['figure.constrained_layout.use'] = True
    print("Read this text.")
    plt.style.use("codes.styles")