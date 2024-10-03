import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import dirname, join
# from utils import *
def fun(x):
    return 200 * (0.2 * (x-0.5) ** 2)# - 0.5 * x)


n = 4
N = 2 ** n
h = 1 / N
x = np.linspace(0, 1, N)

H = np.diag(fun(x))
sns.set_theme(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette("vlag", as_cmap=True)
mask = (H == 0)
# plt.colorbar().remove()
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(H, mask=mask, cmap=cmap, vmax=4, vmin=-4, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
# plt.show()
plt.savefig(join("../../figures/", "diagonal_matrix.pdf"), transparent=True)



'''constant coefficient differential operator'''
H = np.zeros((N,N))
for i in range(N):
    H[i,(i+1)%N] = 1
    H[(i+1)%N,i] = -1
H /= (2 * h)
sns.set_theme(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette("vlag", as_cmap=True)
mask = (H == 0)
# plt.colorbar().remove()
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(H, mask=mask, cmap=cmap, vmax=1/h, vmin=-1/h, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
# plt.show()
plt.savefig(join("../../figures/", "tridiagonal_matrix_homogeneous.pdf"), transparent=True)

'''variable coefficient differential operator'''
H = np.zeros((N,N))
for i in range(N):
    H[i,(i+1)%N] = fun(x[i])
    H[(i+1)%N,i] = -fun(x[i])
H /= (2 * h)
sns.set_theme(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette("vlag", as_cmap=True)
mask = (H == 0)
# plt.colorbar().remove()
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(H, mask=mask, cmap=cmap, vmax=2/h, vmin=-2/h, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
# plt.show()
plt.savefig(join("../../figures/", "tridiagonal_matrix_inhomogeneous.pdf"), transparent=True)
