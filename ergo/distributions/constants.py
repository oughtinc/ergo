import jax.numpy as np

# The default number of points PointDensity uses to represent a distribution
point_density_default_num_points = 200
bin_sizes = np.full(
    point_density_default_num_points, 1 / point_density_default_num_points
)
grid = np.linspace(0, 1, point_density_default_num_points + 1)
target_xs = (grid[1:] + grid[:-1]) / 2
