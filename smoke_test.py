"""Quick end-to-end check of the scaffold on the local monthly CHIRPS subset."""
import numpy as np, torch
from torch.utils.data import DataLoader
from raindeer.data import load_precip, split_years, PrecipPatches
from raindeer import graphs, baselines as B, metrics as M

F_, P = 5, 80  # 0.05deg -> 0.25deg, 80x80 fine patches (16x16 coarse)
da = load_precip("chirps_india_subset.nc")
sp = split_years(da)
print({k: v.sizes["time"] for k, v in sp.items()}, "domain", dict(da.sizes))

tr = PrecipPatches(sp["train"], F_, P, n_samples=64, min_wet_frac=0.2, seed=0)
te = PrecipPatches(sp["test"], F_, P, n_samples=32, min_wet_frac=0.2, seed=1, deterministic=True)
b = next(iter(DataLoader(tr, batch_size=16)))
print("batch", {k: tuple(v.shape) for k, v in b.items()})

# graphs
c = P // F_
g_geo = graphs.geometric_edges(c, c, k=8)
i, j = b["origin"][0, 1].item(), b["origin"][0, 2].item()
ser = graphs.coarse_series_for_patch(sp["train"], i, j, P, F_, n_time=200)
g_cor, w = graphs.corr_edges(ser, k=8)
g_ev, _ = graphs.event_sync_edges(ser, k=8)  # daily-only in practice; shape check here
g_dec, rel = graphs.fine_to_coarse_edges(P, F_, k=4)
print("edges geo/corr/event/decoder:", g_geo.shape, g_cor.shape, g_ev.shape, g_dec.shape, rel.shape)

# baselines
def evalb(fn):
    out = {"rmse": [], "ext": [], "cons": []}
    for bt in DataLoader(te, batch_size=16):
        p = fn(bt["coarse"])
        out["rmse"].append(M.rmse(p, bt["fine"], bt["mask"]))
        out["ext"].append(M.extreme_rmse(p, bt["fine"], mask=bt["mask"]))
        out["cons"].append(M.conservation_error(p, bt["coarse"], F_))
    return {k: round(float(np.nanmean(v)), 3) for k, v in out.items()}

print("nearest ", evalb(lambda c: B.interp(c, F_, "nearest")))
print("bilinear", evalb(lambda c: B.interp(c, F_, "bilinear")))
print("bilin+cons", evalb(lambda c: B.conserve(B.interp(c, F_), c, F_)))
ups = [B.interp(x["coarse"][None], F_)[0].numpy() for x in (tr[k] for k in range(32))]
fns = [tr[k]["fine"].numpy() for k in range(32)]
qm = B.QuantileMapping().fit(ups, fns)
print("qmap+cons", evalb(lambda c: qm.apply(B.interp(c, F_), c, F_)))
net = B.UNetSR()
print("unet fwd", tuple(net.predict_mm(b["coarse"], F_).shape), "(untrained)")
