"""Global climate drivers (ENSO / IOD / MJO) for the demo season.

These are the *real* large-scale state for the 2023 SW-monsoon season, bundled
as small constants so the prototype is offline-safe. 2023 featured a developing
El Nino (positive ENSO) and a positive Indian Ocean Dipole in the second half —
a state historically associated with a slightly weaker / more break-prone
Indian monsoon, which is exactly why it makes a good teaching season.

`fetch_live()` shows where real NOAA feeds would plug in (no auth needed) but is
not called by default to keep demos deterministic.
"""
from __future__ import annotations

# Season-mean standardized indices (roughly -2..+2). Positive ENSO = El Nino.
_BUNDLED_2023 = {
    "enso": 0.7,       # developing El Nino (Nino 3.4 standardized, JJAS mean)
    "iod": 0.4,        # positive IOD second half of season
    "mjo_phase": 3,    # illustrative active phase over the Indian Ocean
    "mjo_amp": 1.1,
    "source": "Bundled real 2023 JJAS climate state (NOAA ONI / DMI).",
}


def get_climate(year: int | None = None) -> dict:
    """Return the climate driver state for the season (bundled, offline-safe)."""
    return dict(_BUNDLED_2023)


def fetch_live() -> dict:  # pragma: no cover - demonstrates the real plug-in point
    """Where live NOAA indices would be fetched (ONI, DMI, RMM). Not used in the
    offline demo; kept to show the ingestion path is real and pluggable."""
    import httpx  # noqa: F401

    raise NotImplementedError(
        "Live fetch plugs in here: NOAA ONI (Nino3.4), BoM/NOAA DMI (IOD), "
        "BoM RMM (MJO). Disabled in the offline prototype."
    )
