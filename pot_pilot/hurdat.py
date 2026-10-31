"""NOAA's HURDAT2 Atlantic hurricane tracks, used by steps 22 and 25.

HURDAT2 (nhc.noaa.gov/data) has a header line per storm (id, name, number of points) and then
one line per track point: date, time (UTC), record type, status, latitude, longitude, wind (kt),
and more. Points are given every 6 hours.
"""
import numpy as np
import pandas as pd

TROPICAL = {"TD", "TS", "HU", "SD", "SS"}    # depression, storm, hurricane, and subtropical


def read_hurdat2(path, first, last):
    """Every track point between two dates, interpolated to every hour.

    The status of each hour is the status at the point before it.
    """
    points, storm, name = [], None, None
    for line in open(path, encoding="utf-8"):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 7:
            storm, name = fields[0], fields[1]
            continue
        lat = float(fields[4][:-1]) * (1 if fields[4][-1] == "N" else -1)
        lon = float(fields[5][:-1]) * (-1 if fields[5][-1] == "W" else 1)
        time_ = pd.Timestamp(fields[0]) + pd.Timedelta(hours=int(fields[1][:2]),
                                                        minutes=int(fields[1][2:]))
        points.append((storm, name, time_, fields[3], lat, lon))
    tracks = pd.DataFrame(points, columns=["storm", "name", "time", "status", "lat", "lon"])
    tracks = tracks[(tracks.time >= first) & (tracks.time <= last)]

    hourly = []
    for storm, track in tracks.groupby("storm"):
        track = track.drop_duplicates("time").set_index("time")
        hours = pd.date_range(track.index[0], track.index[-1], freq="h")
        both = track.index.union(hours)
        part = track[["lat", "lon"]].reindex(both).interpolate("time").loc[hours]
        part["status"] = track.status.reindex(both).ffill().loc[hours]
        part["storm"], part["name"] = storm, track["name"].iloc[0]
        hourly.append(part.reset_index(names="time"))
    return pd.concat(hourly, ignore_index=True)


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine), in km."""
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
