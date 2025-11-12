import earthkit.data as ekd
from earthkit.climate.indicators.precipitation import daily_precipitation_intensity

ek_data = ekd.from_source(
    "url",
    "https://sites.ecmwf.int/repository/earthkit-climate/pr_gridded_day_CMIP6_ACCESS-CM2_r1i1p1f1_deepESD_day_ssp585.nc",
)
daily_precipitation_intensity(ek_data)
