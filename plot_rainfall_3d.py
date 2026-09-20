"""
3D space-time visualization of rainfall (lat, lon, time) using Mayavi.

Plots wet grid points as a 3D point cloud inside a bounding box: the
horizontal plane is lon/lat, the vertical axis is time (day index within
the chosen 3-month window). Point color encodes rain intensity; dry points
are dropped entirely -- deliberately, since rainfall is intermittent, so
the box should mostly look empty with visible clusters where/when it
actually rained. That emptiness is itself informative, not a rendering gap.

Assumes `data` is an xarray Dataset with a `precip` variable and
dims (time, latitude, longitude), as in your earlier CHIRPS snippets.
"""

import numpy as np
from mayavi import mlab
import xarray as xr

data = xr.open_dataset("RainDeer/chirps_india_subset.nc")  # or your own data
# --- 1. Subset: India domain, 3-month window (edit as needed) ---
india = data.sel(latitude=slice(5, 40), longitude=slice(60, 100))
window = india.sel(time=slice('2025-05-01', '2025-10-01'))  # JJA example
precip = window.precip.transpose('time', 'latitude', 'longitude')

lat = precip['latitude'].values
lon = precip['longitude'].values
time = precip['time'].values
values = precip.values  # shape: (n_time, n_lat, n_lon)

n_time, n_lat, n_lon = values.shape
print(f"Grid: {n_time} time steps x {n_lat} lat x {n_lon} lon "
      f"= {n_time * n_lat * n_lon:,} cells")

# --- 2. Threshold to wet points only ---
# Plotting every grid cell (including all the zero/near-zero ones) gives
# millions of overlapping points and a solid, unreadable blob. Keep only
# cells above a wet-day threshold; raise it if the cloud is still too dense
# (e.g. at full 0.05 degree CHIRPS resolution), lower it if it's too sparse
# (e.g. if `data` is already monthly totals rather than daily).
wet_threshold = 200.0  # mm
i_idx, j_idx, k_idx = np.where(values > wet_threshold)
intensity = values[i_idx, j_idx, k_idx]
print(f"Plotting {len(intensity):,} wet points "
      f"({100 * len(intensity) / values.size:.2f}% of the grid)")

# --- 3. Map indices to a physical-looking 3D box ---
# Longitude/latitude are in degrees (tens), time is a day-count (~90) --
# plotting index coordinates keeps axes on comparable scales so one axis
# doesn't visually dwarf the others; mlab.axes() below adds real lon/lat
# /date labels on top of these index coordinates.
x = lon[k_idx]                     # longitude
y = lat[j_idx]                     # latitude
z = i_idx.astype(float)            # day index within the window

# --- 4. Plot ---
mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))

pts = mlab.points3d(
    x, y, z, intensity,
    colormap='RdBu',  # blue=low, red=high
    scale_mode='none',   # fixed point size; only color encodes intensity
    scale_factor=0.25,
    mode='sphere',
)
mlab.colorbar(pts, title='Rain (mm/day)', orientation='vertical')

extent = [lon.min(), lon.max(), lat.min(), lat.max(), 0, n_time - 1]
mlab.outline(extent=extent)
mlab.axes(
    xlabel='Longitude (deg E)',
    ylabel='Latitude (deg N)',
    zlabel='Day of window',
    extent=extent,
    ranges=extent,
)
mlab.title('Rainfall over India: lat / lon / time', height=0.95)

mlab.show()

# --- Notes ---
# - z is a day *index* (0..n_time-1), not a calendar date; print
#   `time[int(z_value)]` to look up the actual date for any point you
#   click on.
# - If nothing renders in Jupyter: call `mlab.init_notebook()` once before
#   the figure block above, or run this as a plain .py script instead --
#   without init_notebook(), Mayavi opens a separate desktop GUI window
#   rather than embedding inline.
# - Alternative worth trying if the point cloud looks too sparse/dense at
#   any single threshold: volume rendering via
#   `mlab.pipeline.volume(mlab.pipeline.scalar_field(values))`, which
#   treats rainfall as a continuous 3D density instead of discrete points.
#   Ask if you want that version -- it needs an opacity transfer function
#   tuned so dry regions stay transparent, which is a bit more setup.