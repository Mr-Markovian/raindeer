"""ERA5 coarse predictors (needs ~/.cdsapirc with a CDS account; not run automatically).

    python -m raindeer.era5 --out data/era5 --years 1981 2025

Pulls daily-mean JJAS fields on the 0.25 deg India box: total column water vapour,
850 hPa u/v wind, 700 hPa relative humidity, CAPE. Verify variable names against the CDS
catalogue ('ERA5 hourly data on single/pressure levels') before a large request.
"""
import argparse
import os

import numpy as np
import xarray as xr

AREA = [38, 66, 6, 99]  # N, W, S, E


def request_year(year, out_dir, months=(6, 7, 8, 9)):
    import cdsapi
    c = cdsapi.Client()
    base = dict(product_type="reanalysis", year=str(year), month=[f"{m:02d}" for m in months],
                day=[f"{d:02d}" for d in range(1, 32)], time=[f"{h:02d}:00" for h in range(0, 24, 3)],
                area=AREA, format="netcdf")
    sfc = os.path.join(out_dir, f"era5_sfc_{year}.nc")
    pl = os.path.join(out_dir, f"era5_pl_{year}.nc")
    if not os.path.exists(sfc):
        c.retrieve("reanalysis-era5-single-levels",
                   dict(base, variable=["total_column_water_vapour", "convective_available_potential_energy",
                                        "total_precipitation"]), sfc)
    if not os.path.exists(pl):
        c.retrieve("reanalysis-era5-pressure-levels",
                   dict(base, variable=["u_component_of_wind", "v_component_of_wind", "relative_humidity"],
                        pressure_level=["700", "850"]), pl)


def to_daily_predictors(sfc_path, pl_path):
    """Daily means stacked as (time, channel, lat, lon); channel names in .attrs['channels']."""
    s = xr.open_dataset(sfc_path).resample(time="1D").mean()
    p = xr.open_dataset(pl_path).resample(time="1D").mean()
    chans = {
        "tcwv": s["tcwv"], "cape": s["cape"],
        "u850": p["u"].sel(pressure_level=850), "v850": p["v"].sel(pressure_level=850),
        "rh700": p["r"].sel(pressure_level=700),
    }
    da = xr.concat(list(chans.values()), dim="channel").assign_coords(channel=list(chans)).transpose(
        "time", "channel", "latitude", "longitude")
    da.attrs["channels"] = list(chans)
    return da


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/era5")
    ap.add_argument("--years", nargs=2, type=int, default=[1981, 2025])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for y in range(a.years[0], a.years[1] + 1):
        request_year(y, a.out)
