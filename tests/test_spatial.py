"""Smoke tests for spatial.py: edge-distance and particle-density QC."""

import pandas as pd


def test_particle_density_outliers_flags_sparse_micrograph():
    from relion_origin.spatial import particle_density_outliers

    rows = []
    for i in range(5):
        rows += [{"rlnMicrographName": "mic%d.mrc" % i}] * (195 + i * 3)
    rows += [{"rlnMicrographName": "mic_bad.mrc"}] * 3
    df = pd.DataFrame(rows)

    density = particle_density_outliers(df)
    assert density.loc["mic_bad.mrc", "_density_outlier"]
    assert not density.loc["mic0.mrc", "_density_outlier"]


def test_flag_near_edge():
    from relion_origin.spatial import flag_near_edge

    df = pd.DataFrame({"_edgeDistanceMin_angst": [1000, 1050, 980, 1020, 990, 50]})
    flagged = flag_near_edge(df)
    assert flagged.iloc[-1]
    assert not flagged.iloc[:-1].any()


def test_edge_distance_sta():
    from relion_origin.spatial import edge_distance_sta

    particles = pd.DataFrame(
        {
            "rlnCoordinateX": [10.0, 200.0],
            "rlnCoordinateY": [10.0, 200.0],
            "rlnCoordinateZ": [10.0, 100.0],
            "rlnMicrographName": ["tomoA.tomostar", "tomoA.tomostar"],
            "rlnPixelSize": [10.0, 10.0],
        }
    )
    dims = pd.DataFrame(
        {"rlnTomoSizeX": [4000], "rlnTomoSizeY": [4000], "rlnTomoSizeZ": [2000], "rlnTomoTiltSeriesPixelSize": [1.0]},
        index=pd.Index(["tomoA.tomostar"], name="rlnTomoName"),
    )
    result = edge_distance_sta(particles, dims)
    # particle 0 near the corner (10px * 10A/px = 100A from origin, tomogram is 4000A wide)
    assert result.loc[0, "_edgeDistanceMin_angst"] < result.loc[1, "_edgeDistanceMin_angst"]


def test_edge_distance_spa():
    from relion_origin.spatial import edge_distance_spa

    particles = pd.DataFrame(
        {
            "rlnCoordinateX": [10.0, 2000.0, 3990.0],
            "rlnCoordinateY": [10.0, 2000.0, 10.0],
            "rlnMicrographName": ["mic1.mrc", "mic1.mrc", "mic1.mrc"],
            "rlnPixelSize": [1.0, 1.0, 1.0],
        }
    )
    dims = pd.DataFrame({"SizeX": [4000], "SizeY": [4000]}, index=pd.Index(["mic1.mrc"], name="rlnMicrographName"))

    result = edge_distance_spa(particles, dims)
    assert result.loc[0, "_edgeDistanceMin_angst"] == 10.0
    assert result.loc[2, "_edgeDistanceMin_angst"] == 10.0
    assert result.loc[1, "_edgeDistanceMin_angst"] == 2000.0


def test_edge_distance_spa_missing_micrograph_raises():
    from relion_origin.spatial import edge_distance_spa

    particles = pd.DataFrame({"rlnCoordinateX": [1.0], "rlnCoordinateY": [1.0], "rlnMicrographName": ["unknown.mrc"]})
    dims = pd.DataFrame({"SizeX": [4000], "SizeY": [4000]}, index=pd.Index(["mic1.mrc"], name="rlnMicrographName"))
    try:
        edge_distance_spa(particles, dims)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a micrograph missing from dims_df")


if __name__ == "__main__":
    test_particle_density_outliers_flags_sparse_micrograph()
    test_flag_near_edge()
    test_edge_distance_sta()
    test_edge_distance_spa()
    test_edge_distance_spa_missing_micrograph_raises()
    print("spatial tests passed")
