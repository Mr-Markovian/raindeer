"""
A comparable topological "signature" for a rainfall field -- the
topological analogue of the universal-multifractal parameters (alpha, C1,
H) from the multifractal downscaling work. Same motivation: instead of
picking one persistence threshold and extracting one discrete skeleton
(fragile, and rain's own scale-free structure may not even give you a
clean gap to threshold at), compute a small set of STABLE, FIXED-RANGE
descriptors of the entire persistence diagram. Two different days,
regions, or an observed-vs-generated pair can then be compared directly
by comparing their signatures, with no threshold-picking anywhere.

Depends on merge_tree_rainfall.py for the persistence computation and the
optional morphological denoiser.

THE ONE RULE THAT MAKES THIS A "SIGNATURE" RATHER THAN JUST A PLOT:
every field you want to compare must be vectorized on the SAME fixed
intensity range (`max_intensity` below), decided once from domain
knowledge (e.g. "daily monsoon rainfall over this region rarely exceeds
150mm"), never re-derived per-field. If you let each field set its own
range, the bins mean different things for different days and the
resulting vectors are not comparable -- which defeats the entire point.
"""

import numpy as np
import matplotlib.pyplot as plt
import gudhi.representations as gr
import xarray as xr

from merge_tree_rainfall import superlevel_persistence_gudhi


def to_gudhi_diagram(births, deaths):
    """
    Our persistence values come out as (birth, death) in real, superlevel
    units (birth = peak intensity, death = intensity where it merged away,
    so birth > death). gudhi.representations expects the opposite,
    standard sublevel convention (birth <= death). This does that
    conversion -- both sides represent the exact same persistence
    (|birth - death| is unchanged either way), only the sign/order flips.
    """
    finite = np.isfinite(deaths)
    b, d = births[finite], deaths[finite]
    return np.column_stack([-b, -d])  # now satisfies birth' <= death'


def compute_signature(field, min_value=1.0, max_intensity=150.0,
                       resolution=100, num_landscapes=3):
    """
    field: 2D array (already denoised if you're using
        merge_tree_rainfall.denoise_morphological -- that's a choice made
        before this function, not inside it).
    min_value: filtration floor, same meaning as in build_merge_tree.
    max_intensity: the FIXED upper end of the comparison range, in the
        same units as the field (mm). Must be the same value for every
        field you intend to compare against each other. Pick it from
        domain knowledge (a sensible cap on daily rainfall for your
        region/season), not from this field's own max.

    Returns a dict of numpy arrays: betti_curve, landscape,
    persistence_image, entropy (scalar), persistence_exponent (scalar,
    see note below), plus the raw diagram for inspection.
    """
    births, deaths = superlevel_persistence_gudhi(field, min_value=min_value)
    if births is None:
        raise ImportError("gudhi is required for compute_signature")

    diag = to_gudhi_diagram(births, deaths)
    # Negated sample range: real intensities live in [min_value, max_intensity],
    # so in the sign-flipped GUDHI convention that's [-max_intensity, -min_value].
    sample_range = [-max_intensity, -min_value]

    if diag.shape[0] == 0:
        # Nothing survived the filtration floor at all -- report an
        # all-zero signature rather than letting the vectorizers choke on
        # an empty diagram, and say so loudly, since it usually means
        # min_value is set too high for this particular field.
        print("WARNING: no finite persistence pairs -- min_value may be "
              "too high for this field, or the field is essentially dry.")
        betti = np.zeros(resolution)
        landscape = np.zeros(resolution * num_landscapes)
        pimage = np.zeros(400)  # matches default 20x20 below
        entropy = 0.0
        exponent = np.nan
    else:
        betti = gr.BettiCurve(resolution=resolution,
                               sample_range=sample_range).fit_transform([diag])[0]
        landscape = gr.Landscape(num_landscapes=num_landscapes, resolution=resolution,
                                  sample_range=sample_range).fit_transform([diag])[0]
        pimage = gr.PersistenceImage(
            resolution=[20, 20],
            im_range=sample_range + sample_range,
        ).fit_transform([diag])[0]
        entropy = gr.Entropy(mode='scalar').fit_transform([diag])[0, 0]
        exponent = persistence_exponent(births[np.isfinite(deaths)] -
                                         deaths[np.isfinite(deaths)])

    return dict(betti_curve=betti, landscape=landscape,
                persistence_image=pimage, entropy=entropy,
                persistence_exponent=exponent,
                births=births, deaths=deaths,
                sample_range=sample_range, min_value=min_value,
                max_intensity=max_intensity)


def persistence_exponent(persistences, min_points=5):
    """
    Fits a power law to the rank-ordered persistence spectrum: sort
    persistence values descending, regress log(rank) on log(persistence).
    The slope is a single scalar describing how quickly features fall off
    in importance -- if rainfall's cascade structure is genuinely
    scale-free, this spectrum is expected to look roughly power-law rather
    than having a clean two-population (noise vs. signal) shape, which is
    the honest reason your persistence diagram may never show a clean gap
    to threshold at. This exponent is the closest topological analogue to
    the multifractal moment-scaling exponents from the downscaling work --
    treat it as exploratory, not an established literature quantity; I
    have not verified this specific construction against published work,
    so validate it against synthetic multifractal cascades with known
    exponents before reading much into absolute values.
    """
    p = np.sort(np.asarray(persistences))[::-1]
    p = p[p > 0]
    if len(p) < min_points:
        return np.nan
    rank = np.arange(1, len(p) + 1)
    slope, _ = np.polyfit(np.log(p), np.log(rank), 1)
    return float(slope)


def plot_signature(sig, title=''):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(sig['min_value'], sig['max_intensity'],
                     len(sig['betti_curve']))
    axes[0].plot(x, sig['betti_curve'])
    axes[0].set_xlabel('intensity threshold (mm)')
    axes[0].set_ylabel('# live components (Betti-0)')
    axes[0].set_title('Betti curve')

    n_land = len(sig['landscape']) // (len(x))
    land = sig['landscape'].reshape(n_land, -1) if n_land else sig['landscape'].reshape(1, -1)
    xl = np.linspace(sig['min_value'], sig['max_intensity'], land.shape[1])
    for i in range(land.shape[0]):
        axes[1].plot(xl, land[i], label=f'level {i}')
    axes[1].set_xlabel('intensity threshold (mm)')
    axes[1].set_title('Persistence landscape')
    axes[1].legend(fontsize=7)

    finite = np.isfinite(sig['deaths'])
    pers = sig['births'][finite] - sig['deaths'][finite]
    pers = np.sort(pers[pers > 0])[::-1]
    if len(pers) > 0:
        rank = np.arange(1, len(pers) + 1)
        axes[2].loglog(pers, rank, 'o-', ms=3)
        axes[2].set_xlabel('persistence (mm, log scale)')
        axes[2].set_ylabel('rank (log scale)')
        axes[2].set_title(f'Persistence spectrum (slope={sig["persistence_exponent"]:.2f})')

    fig.suptitle(f"{title}  |  entropy={sig['entropy']:.3f}")
    plt.tight_layout()
    return fig


def signature_distance(sig1, sig2, which='landscape'):
    """
    L2 distance between two signatures' chosen vector representation.
    This is the actual point of building a fixed-range, comparable
    signature: you can now say "day A and day B differ by X" or "the
    generated field's topology is Y away from the observed field's,"
    which a hand-thresholded skeleton never lets you say cleanly.
    """
    v1, v2 = sig1[which], sig2[which]
    return float(np.linalg.norm(v1 - v2))


if __name__ == '__main__':
    # Two different days from the same region, compared directly --
    # this is the actual workflow: build the signature once per field,
    # then compare, rather than eyeballing two skeletons side by side.
    data = xr.open_dataset("RainDeer/chirps_india_subset.nc")  # or your own data
    region = data.sel(latitude=slice(8, 21), longitude=slice(73, 77))
    day1 = region.precip.sel(time='2025-07-01').values
    day2 = region.precip.sel(time='2025-08-01').values

    MIN_VALUE = 1.0
    MAX_INTENSITY = 150.0  # fix this once from domain knowledge; reuse everywhere

    sig1 = compute_signature(day1, min_value=MIN_VALUE, max_intensity=MAX_INTENSITY)
    sig2 = compute_signature(day2, min_value=MIN_VALUE, max_intensity=MAX_INTENSITY)

    print(f"Day 1: entropy={sig1['entropy']:.3f}, "
          f"persistence exponent={sig1['persistence_exponent']:.3f}")
    print(f"Day 2: entropy={sig2['entropy']:.3f}, "
          f"persistence exponent={sig2['persistence_exponent']:.3f}")
    print(f"Landscape distance between the two days: "
          f"{signature_distance(sig1, sig2, 'landscape'):.3f}")

    plot_signature(sig1, title='2025-07-15')
    plt.show()
    plot_signature(sig2, title='2025-07-20')
    plt.show()