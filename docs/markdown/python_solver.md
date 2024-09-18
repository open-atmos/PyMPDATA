<details>
<summary>Python code (click to expand)</summary>

```Python
def plot(psi, zlim, norm=None):
    xi, yi = np.indices(psi.shape)
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    pyplot.gca().plot_wireframe(
        xi+.5, yi+.5,
        psi, color='red', linewidth=.5
    )
    ax.set_zlim(zlim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('black')
        axis.pane.set_alpha(1)
    ax.grid(False)
    ax.set_zticks([])
    ax.set_xlabel('x/dx')
    ax.set_ylabel('y/dy')
    ax.set_proj_type('ortho')
    cnt = ax.contourf(xi+.5, yi+.5, psi, zdir='z', offset=-1, norm=norm)
    cbar = pyplot.colorbar(cnt, pad=.1, aspect=10, fraction=.04)
    return cbar.norm

zlim = (-1, 1)
norm = plot(state_0, zlim)
pyplot.savefig('readme_gauss_0.png')
plot(state, zlim, norm)
pyplot.savefig('readme_gauss.png')
```
</details>

![plot](https://github.com/open-atmos/PyMPDATA/releases/download/tip/readme_gauss_0.png)
![plot](https://github.com/open-atmos/PyMPDATA/releases/download/tip/readme_gauss.png)
