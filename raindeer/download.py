"""Download CHIRPS v3 daily, cropped to a bbox, one NetCDF per year (resumable).

CHIRPS v3 daily is one global GeoTIFF per day (~17 MB). We read only the bbox window over
HTTP (/vsicurl/), so no global files are stored. Expect ~5 s/day/thread; JJAS-only for
1981-2025 is ~5.5k days (use --workers 8).

    python -m raindeer.download --out data/chirps_daily --years 1981 2025 --months 6 7 8 9
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import from_bounds

BASE = "/vsicurl/https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/{y}/chirps-v3.0.sat.{y}.{m:02d}.{d:02d}.tif"
BBOX = (6.0, 38.0, 66.0, 99.0)  # lat0, lat1, lon0, lon1


def read_day(day, bbox=BBOX, retries=3):
    la0, la1, lo0, lo1 = bbox
    url = BASE.format(y=day.year, m=day.month, d=day.day)
    for k in range(retries):
        try:
            with rasterio.open(url) as s:
                win = from_bounds(lo0, la0, lo1, la1, s.transform)
                a = s.read(1, window=win)
                tr = s.window_transform(win)
            lat = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5)  # north->south
            lon = tr.c + tr.a * (np.arange(a.shape[1]) + 0.5)
            return a.astype(np.float32), lat, lon
        except Exception:
            if k == retries - 1:
                raise
    

def download_year(year, out_dir, months, workers=8, bbox=BBOX):
    path = os.path.join(out_dir, f"chirps_v3_daily_{year}.nc")
    if os.path.exists(path):
        return path
    days = [d for d in (date(year, 1, 1) + timedelta(i) for i in range(366))
            if d.year == year and d.month in months]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda d: read_day(d, bbox), days))
    arr = np.stack([r[0] for r in res])
    arr[arr < 0] = np.nan  # -9999 fill
    lat, lon = res[0][1], res[0][2]
    da = xr.DataArray(arr, dims=("time", "latitude", "longitude"),
                      coords={"time": np.array(days, dtype="datetime64[ns]"),
                              "latitude": lat, "longitude": lon}, name="precip")
    da = da.sortby("latitude")
    tmp = path + ".part"
    da.to_dataset().to_netcdf(tmp, encoding={"precip": {"zlib": True, "complevel": 4}})
    os.replace(tmp, path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chirps_daily")
    ap.add_argument("--years", nargs=2, type=int, default=[1981, 2025])
    ap.add_argument("--months", nargs="+", type=int, default=[6, 7, 8, 9])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for y in range(a.years[0], a.years[1] + 1):
        print(y, download_year(y, a.out, set(a.months), a.workers), flush=True)
