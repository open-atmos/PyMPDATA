<details>
<summary>Python code (click to expand)</summary>

```Python
import numpy as np
from matplotlib import pyplot

dx, dy = .2, .3
grid = (10, 5)

pyplot.scatter(*np.mgrid[
        dx / 2 : grid[0] * dx : dx,
        dy / 2 : grid[1] * dy : dy
    ], color='red',
    label='scalar-field values at cell centres'
)
pyplot.quiver(*np.mgrid[
        0 : (grid[0]+1) * dx : dx,
        dy / 2 : grid[1] * dy : dy
    ], 1, 0, pivot='mid', color='green', width=.005,
    label='vector-field x-component values at cell walls'
)
pyplot.quiver(*np.mgrid[
        dx / 2 : grid[0] * dx : dx,
        0: (grid[1] + 1) * dy : dy
    ], 0, 1, pivot='mid', color='blue', width=.005,
    label='vector-field y-component values at cell walls'
)
pyplot.xticks(np.linspace(0, grid[0]*dx, grid[0]+1))
pyplot.yticks(np.linspace(0, grid[1]*dy, grid[1]+1))
pyplot.title(f'staggered grid layout (grid={grid}, dx={dx}, dy={dy})')
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.legend(bbox_to_anchor=(.1, -.1), loc='upper left', ncol=1)
pyplot.grid()
pyplot.savefig('readme_grid.png')
```
</details>
