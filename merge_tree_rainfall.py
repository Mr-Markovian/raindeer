"""
Merge-tree skeleton extraction for rainfall fields -- no pre-smoothing.

Implements the superlevel-set merge tree directly via union-find (the
standard "elder rule" construction from computational topology), so the
only hard dependency is numpy/xarray/networkx/matplotlib. GUDHI is used
as an independent cross-check on the persistence values, not as the
skeleton-builder itself -- this keeps the mechanism visible rather than
hiding it inside a library call, and sidesteps TTK's heavier VTK-based
install if you just want to get moving.

Pipeline, matching what we agreed on:
  1. Work at native resolution on a subregion first. No smoothing.
  2. Build the FULL merge tree (every local rain maximum is a leaf).
  3. Look at the persistence diagram to see the birth/death distribution
     and choose a simplification threshold -- this replaces smoothing as
     the noise-handling step (short-lived branches = noise, by
     construction, not by an arbitrary kernel bandwidth).
  4. Prune the tree at that threshold: this is your skeleton.
  5. Repeat at a couple of coarse-graining levels and compare -- branches
     that survive coarsening are real; branches that don't are
     resolution artifacts. This sweep is itself a scaling diagnostic,
     complementary to the K(q) multifractal analysis.
"""

import warnings
import numpy as np
import xarray as xr
import networkx as nx
import matplotlib.pyplot as plt

try:
    import gudhi
    HAVE_GUDHI = True
except ImportError:
    HAVE_GUDHI = False
    print("gudhi not installed (pip install gudhi) -- skipping the "
          "persistence-diagram cross-check; the merge tree still works.")

try:
    from scipy.ndimage import grey_opening
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# 0. Denoising -- a better-justified alternative to Gaussian smoothing
# ---------------------------------------------------------------------------

def denoise_morphological(field, size=3):
    """
    Removes noise-driven local maxima narrower than `size` pixels while
    leaving genuinely storm-scale peaks' amplitudes essentially untouched.

    This is a different (and for this specific problem, better justified)
    tool than Gaussian smoothing: a Gaussian blur reduces EVERY peak's
    amplitude, real or not, which biases persistence values downward
    across the board. Grayscale morphological opening (erosion then
    dilation) instead removes features that are narrower than the
    structuring element and leaves wider, genuinely-supported peaks
    numerically close to unchanged. It is still a real modeling choice --
    `size` must be strictly smaller than the storm-cell scale you actually
    care about, or you will erase real small storms along with the noise.
    Always compare against the undenoised version; don't take this on
    faith.

    Known limitation: cells adjacent to a NaN mask can be pulled down by
    the fill value during opening. Fine for interior noise removal; treat
    values within `size` pixels of a mask boundary with more caution.
    """
    if not HAVE_SCIPY:
        raise ImportError("pip install scipy for denoise_morphological")
    valid = np.isfinite(field)
    filled = np.where(valid, field, -np.inf)
    opened = grey_opening(filled, size=size)
    return np.where(valid, opened, np.nan)


# ---------------------------------------------------------------------------
# 1. Merge tree construction (superlevel-set union-find, elder rule)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def make(self, x):
        self.parent[x] = x

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return self.find(a)


def build_merge_tree(field, min_value=1.0, connectivity=4):
    """
    field: 2D array (lat, lon); NaN for masked/missing cells.
    min_value: cells below this are excluded from the filtration entirely.
        Without this, the whole dry background eventually merges into one
        giant component and the tree wastes almost all its structure on
        pixels that were never meaningfully "raining." Raise it if your
        skeleton still looks dominated by drizzle-level noise.
    connectivity: 4 or 8.

    Returns a networkx.Graph. Node attrs: value, row, col,
    kind ('leaf' = local rain maximum, 'merge' = two rain cells joining).
    Edge attr: persistence = birth_value - death_value of the branch that
    terminates at that merge (the standard topological "how real is this
    feature" score).
    """
    n_rows, n_cols = field.shape
    mask = np.isfinite(field) & (field >= min_value)
    rs, cs = np.where(mask)
    order = np.argsort(-field[rs, cs])           # descending: superlevel sets
    rs, cs = rs[order], cs[order]

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    uf = UnionFind()
    tree = nx.Graph()
    processed = np.zeros_like(field, dtype=bool)

    elder_value = {}     # root -> birth value of that branch's peak
    current_node = {}    # root -> tree node id at the current tip of that branch
    next_id = [0]

    def new_node(value, r, c, kind):
        nid = next_id[0]
        next_id[0] += 1
        tree.add_node(nid, value=float(value), row=int(r), col=int(c), kind=kind)
        return nid

    for r, c in zip(rs, cs):
        val = field[r, c]
        idx = (r, c)

        touched_roots = set()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and processed[nr, nc]:
                touched_roots.add(uf.find((nr, nc)))

        uf.make(idx)
        processed[r, c] = True

        if not touched_roots:
            # New local rain maximum: a leaf is born.
            leaf_id = new_node(val, r, c, 'leaf')
            elder_value[idx] = val
            current_node[idx] = leaf_id

        elif len(touched_roots) == 1:
            # Extends an existing storm cell; not a tree event.
            root = next(iter(touched_roots))
            new_root = uf.union(root, idx)
            elder_value[new_root] = elder_value.pop(root, elder_value.get(new_root))
            current_node[new_root] = current_node.pop(root, current_node.get(new_root))

        else:
            # Two or more storm cells touch here: a merge event.
            roots = list(touched_roots)
            elder_root = max(roots, key=lambda rt: elder_value[rt])
            merge_id = new_node(val, r, c, 'merge')

            for rt in roots:
                if rt != elder_root:
                    persistence = elder_value[rt] - val
                    tree.add_edge(current_node[rt], merge_id,
                                  persistence=float(persistence))

            # The elder's own branch doesn't die here -- it continues
            # through this point. Still needs an edge, or the tree ends up
            # as a disconnected forest instead of one connected skeleton:
            # persistence isn't "decided" yet (this branch could still die
            # later at a lower merge), so mark it as a non-prunable
            # continuation rather than assigning it a false finite value.
            tree.add_edge(current_node[elder_root], merge_id,
                          persistence=float('inf'), is_continuation=True)

            for rt in roots:
                uf.union(idx, rt)
            final_root = uf.find(idx)

            elder_value[final_root] = elder_value[elder_root]
            current_node[final_root] = merge_id

    return tree


# ---------------------------------------------------------------------------
# 2. Persistence-based pruning -- this is your "smoothing" replacement
# ---------------------------------------------------------------------------

def prune_tree(tree, persistence_threshold):
    """
    Removes leaves whose branch persistence is below threshold, then
    contracts any merge node left with degree <= 2 (it's no longer a real
    branch point once its short-lived children are gone).
    """
    t = tree.copy()
    changed = True
    while changed:
        changed = False
        leaves = [n for n, d in t.degree() if d == 1 and t.nodes[n]['kind'] == 'leaf']
        for leaf in leaves:
            neighbor = next(iter(t[leaf]))
            if t[leaf][neighbor]['persistence'] < persistence_threshold:
                t.remove_node(leaf)
                changed = True
        # contract pass-through merge nodes (degree 2), and drop merge
        # nodes that pruning left as dead-end stubs (degree <= 1) -- once
        # all its short branches are gone, a merge node with only one
        # remaining edge isn't a real branch point anymore.
        for n in list(t.nodes):
            if n not in t:
                continue
            if t.nodes[n]['kind'] == 'merge' and t.degree(n) == 2:
                nbrs = list(t[n])
                p = max(t[n][nbrs[0]]['persistence'], t[n][nbrs[1]]['persistence'])
                t.remove_node(n)
                t.add_edge(nbrs[0], nbrs[1], persistence=p)
                changed = True
            elif t.nodes[n]['kind'] == 'merge' and t.degree(n) <= 1:
                t.remove_node(n)
                changed = True
            elif t.degree(n) == 0:
                t.remove_node(n)
                changed = True
    return t


# ---------------------------------------------------------------------------
# 3. Independent cross-check via GUDHI (persistence diagram, not the tree)
# ---------------------------------------------------------------------------

def superlevel_persistence_gudhi(field, min_value=1.0):
    """
    Returns (births, deaths) for 0-dimensional superlevel-set persistence,
    computed independently of build_merge_tree above, as a sanity check:
    the number of finite pairs here should match the number of pruned
    leaves you'd get at persistence_threshold=0 in the merge tree.
    GUDHI's CubicalComplex does sublevel filtration, so we negate the
    field to get superlevel persistence, then negate the values back.
    """
    if not HAVE_GUDHI:
        return None, None
    filled = np.where(np.isfinite(field) & (field >= min_value), field, -np.inf)
    cc = gudhi.CubicalComplex(top_dimensional_cells=-filled)
    cc.compute_persistence()
    pairs = cc.persistence_intervals_in_dimension(0)
    births = -pairs[:, 0]
    deaths = np.where(np.isfinite(pairs[:, 1]), -pairs[:, 1], np.nan)
    return births, deaths


# ---------------------------------------------------------------------------
# 4. Coarse-graining for the scaling sweep
# ---------------------------------------------------------------------------

def coarse_grain(field, factor):
    """Block-mean downsampling by an integer factor; trims to fit evenly.

    A block that is entirely NaN (e.g. fully inside a masked/ocean region)
    legitimately stays NaN in the output -- numpy warns about this
    ("Mean of empty slice"), which is expected and silenced here rather
    than treated as an error.
    """
    n_rows, n_cols = field.shape
    n_rows_t = (n_rows // factor) * factor
    n_cols_t = (n_cols // factor) * factor
    trimmed = field[:n_rows_t, :n_cols_t]
    reshaped = trimmed.reshape(n_rows_t // factor, factor, n_cols_t // factor, factor)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(reshaped, axis=(1, 3))


# ---------------------------------------------------------------------------
# 5. Plotting: overlay the skeleton on the actual rain map (not an
#    abstract dendrogram -- this is what makes it a spatial "skeleton")
# ---------------------------------------------------------------------------

def plot_skeleton(field, tree, lat, lon, ax=None, title=''):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(field, origin='lower', cmap='Blues',
              extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              aspect='auto')

    for u, v, data in tree.edges(data=True):
        r1, c1 = tree.nodes[u]['row'], tree.nodes[u]['col']
        r2, c2 = tree.nodes[v]['row'], tree.nodes[v]['col']
        ax.plot([lon[c1], lon[c2]], [lat[r1], lat[r2]], '-', color='black', lw=1)

    leaf_rc = [(tree.nodes[n]['row'], tree.nodes[n]['col'])
               for n in tree.nodes if tree.nodes[n]['kind'] == 'leaf']
    merge_rc = [(tree.nodes[n]['row'], tree.nodes[n]['col'])
                for n in tree.nodes if tree.nodes[n]['kind'] == 'merge']
    if leaf_rc:
        lr, lc = zip(*leaf_rc)
        ax.scatter(lon[list(lc)], lat[list(lr)], c='red', s=25,
                   zorder=3, label='local max (leaf)')
    if merge_rc:
        mr, mc = zip(*merge_rc)
        ax.scatter(lon[list(mc)], lat[list(mr)], c='orange', s=15,
                   zorder=3, label='merge event')

    ax.set_title(title)
    ax.set_xlabel('Longitude (deg E)')
    ax.set_ylabel('Latitude (deg N)')
    ax.legend(loc='upper right', fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Subregion first, native resolution, no smoothing ---
    # Western Ghats box, matching the region already used in the
    # downscaling problem statement. Adjust bounds/date to what you have.
    region = data.sel(latitude=slice(8, 21), longitude=slice(73, 77))
    field_da = region.precip.sel(time='2025-07-15')  # pick a real wet day
    field = field_da.values
    lat = field_da['latitude'].values
    lon = field_da['longitude'].values

    min_value = 1.0  # mm; the "not part of the filtration" floor

    # Step 1: look at the persistence diagram before picking a threshold.
    births, deaths = superlevel_persistence_gudhi(field, min_value=min_value)
    if births is not None:
        finite = np.isfinite(deaths)
        persistence = births[finite] - deaths[finite]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(births[finite], persistence, s=10, alpha=0.6)
        ax.set_xlabel('birth (mm)')
        ax.set_ylabel('persistence (mm)')
        ax.set_title('Persistence of storm cells -- pick your threshold here')
        plt.show()
        print(f"GUDHI found {finite.sum()} finite 0-dim pairs "
              f"(+1 essential class for the whole-field root).")

    # Step 2: build the full tree, then prune at a threshold read off the
    # plot above (start here, then adjust once you've looked at it).
    persistence_threshold = 5.0  # mm
    full_tree = build_merge_tree(field, min_value=min_value)
    pruned = prune_tree(full_tree, persistence_threshold)
    print(f"Full tree: {full_tree.number_of_nodes()} nodes -> "
          f"pruned skeleton: {pruned.number_of_nodes()} nodes "
          f"(threshold = {persistence_threshold} mm)")

    plot_skeleton(field, pruned, lat, lon,
                  title=f'Rainfall skeleton (persistence > {persistence_threshold} mm)')
    plt.show()

    # Step 3: coarse-graining sweep -- is this structure real or resolution?
    print("\nCoarse-graining sweep:")
    for factor in [1, 2, 4]:
        cg_field = coarse_grain(field, factor) if factor > 1 else field
        cg_tree = build_merge_tree(cg_field, min_value=min_value)
        cg_pruned = prune_tree(cg_tree, persistence_threshold)
        n_leaves = sum(1 for n in cg_pruned.nodes
                       if cg_pruned.nodes[n]['kind'] == 'leaf')
        print(f"  factor {factor}x  ({cg_field.shape[0]}x{cg_field.shape[1]} grid): "
              f"{n_leaves} surviving leaves in the pruned skeleton")