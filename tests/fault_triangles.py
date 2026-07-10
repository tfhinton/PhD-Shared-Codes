#%%
from codes import FaultTriangles
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

d = np.load("/Users/hintont/Dev/projects/ridgecrest/data/fault/mainshock_fault_remesh.npz")
fault = FaultTriangles(d["vertices"], d["triangles"], layers=d["layers"])

xs = np.linspace(430000, 475000, 100)
ys = np.linspace(3.935e6, 3.980e6, 100)
xx, yy = np.meshgrid(xs, ys)
xs = xx.ravel()
ys = yy.ravel()
pts = np.stack([xs, ys], axis=0)

fault = fault.compute_greens_functions(pts)
print(fault.gfs)
north_ss_gfs = fault.gfs[0, :, 1, :]
north_ss_gfs = np.sum(north_ss_gfs, axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(north_ss_gfs.reshape(xx.shape), extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", cmap=cmc.vik, interpolation="bilinear")
plt.show()

fault.plot_fault3d(color_by="layer", cmap=cmc.tokyo)
plt.show()