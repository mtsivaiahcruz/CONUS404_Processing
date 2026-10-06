import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import genextreme, gumbel_r

from coordinate_utils import write_hdf5_coordinates


MODULE_PATH = Path(__file__).resolve().parents[1] / "40_calculate_return_periods.py"
SPEC = importlib.util.spec_from_file_location("calculate_return_periods", MODULE_PATH)
RETURN_PERIODS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETURN_PERIODS)


class Convert20sTo3sGustTests(unittest.TestCase):
    def test_scalar_uses_rounded_durst_factor(self):
        converted = RETURN_PERIODS.convert_20s_to_3s_gust(100.0)

        self.assertAlmostEqual(converted, 111.76)
        self.assertEqual(RETURN_PERIODS.convert_20s_to_3s_gust(1.0), 1.1176)
        self.assertNotEqual(converted, 100.0 * (1.52 / 1.36))
        self.assertNotEqual(converted, 100.0 * 1.118)

    def test_array_preserves_shape_dtype_nans_and_input(self):
        native = np.array([[10.0, np.nan], [25.5, 0.0]], dtype=np.float32)
        original = native.copy()

        converted = RETURN_PERIODS.convert_20s_to_3s_gust(native)

        self.assertEqual(converted.shape, native.shape)
        self.assertEqual(converted.dtype, native.dtype)
        np.testing.assert_array_equal(native, original)
        np.testing.assert_allclose(
            converted,
            native * np.float32(1.1176),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


class GevReturnLevelTests(unittest.TestCase):
    def test_zero_shape_matches_gumbel(self):
        for return_period in (10, 100, 500):
            level = RETURN_PERIODS.gev_return_level(return_period, 30.0, 5.0, 0.0)
            expected = gumbel_r.isf(1.0 / return_period, loc=30.0, scale=5.0)

            self.assertAlmostEqual(level, expected, places=9)

    def test_nonzero_shape_matches_scipy(self):
        for shape in (-0.3, -0.05, 0.05, 0.3):
            level = RETURN_PERIODS.gev_return_level(100, 30.0, 5.0, shape)
            expected = genextreme.isf(0.01, shape, loc=30.0, scale=5.0)

            self.assertAlmostEqual(level, expected, places=9)

    def test_return_level_is_continuous_at_zero_shape(self):
        at_zero = RETURN_PERIODS.gev_return_level(100, 30.0, 5.0, 0.0)

        for shape in (-1e-5, 1e-5):
            nearby = RETURN_PERIODS.gev_return_level(100, 30.0, 5.0, shape)
            self.assertAlmostEqual(nearby, at_zero, places=2)


class GevReturnLevelCiTests(unittest.TestCase):
    def test_information_matrix_at_zero_shape(self):
        # Known values for the Gumbel limit of the GEV.
        euler = np.euler_gamma
        information = RETURN_PERIODS.gev_information_matrix(0.0)

        self.assertAlmostEqual(information[0, 0], 1.0, places=3)
        self.assertAlmostEqual(information[0, 1], -(1.0 - euler), places=3)
        self.assertAlmostEqual(
            information[1, 1], np.pi**2 / 6.0 + (1.0 - euler) ** 2, places=3
        )
        np.testing.assert_array_equal(information, information.T)

    def test_information_matrix_matches_average_of_scores(self):
        # The information matrix is the average of score x score', where the
        # score is the derivative of the log density in each parameter. The
        # average is taken over an even grid of probabilities.
        step = 1e-5
        for shape in (-0.2, 0.1):
            probabilities = (np.arange(200000) + 0.5) / 200000
            sample = genextreme.ppf(probabilities, shape)
            parameters = np.array([0.0, 1.0, shape])
            scores = []
            for index in range(3):
                up = parameters.copy()
                down = parameters.copy()
                up[index] += step
                down[index] -= step
                scores.append(
                    (
                        genextreme.logpdf(sample, up[2], up[0], up[1])
                        - genextreme.logpdf(sample, down[2], down[0], down[1])
                    )
                    / (2.0 * step)
                )
            scores = np.array(scores)
            average = scores @ scores.T / sample.size

            np.testing.assert_allclose(
                RETURN_PERIODS.gev_information_matrix(shape), average, atol=0.03
            )

    def test_gradient_matches_numerical_derivative(self):
        step = 1e-5
        for shape in (-0.2, 0.0, 0.2):
            gradient = RETURN_PERIODS.gev_return_level_gradient(100, 5.0, shape)
            parameters = np.array([30.0, 5.0, shape])
            for index in range(3):
                up = parameters.copy()
                down = parameters.copy()
                up[index] += step
                down[index] -= step
                numerical = (
                    genextreme.isf(0.01, up[2], up[0], up[1])
                    - genextreme.isf(0.01, down[2], down[0], down[1])
                ) / (2.0 * step)

                self.assertAlmostEqual(gradient[index], numerical, places=4)

    def test_interval_is_centred_on_the_estimate(self):
        estimate = RETURN_PERIODS.gev_return_level(100, 12.0, 1.5, -0.1)
        lower, upper = RETURN_PERIODS.gev_return_level_ci(
            100, 12.0, 1.5, -0.1, data_points=45
        )

        self.assertGreater(lower, 0.0)
        self.assertLess(lower, estimate)
        self.assertAlmostEqual((lower + upper) / 2.0, estimate)

    def test_interval_widens_with_return_period(self):
        widths = []
        for return_period in (10, 25, 50, 100, 200, 500):
            lower, upper = RETURN_PERIODS.gev_return_level_ci(
                return_period, 12.0, 1.5, -0.1, data_points=45
            )
            widths.append(upper - lower)

        self.assertEqual(widths, sorted(widths))

    def test_four_times_the_years_halves_the_width(self):
        lower_45, upper_45 = RETURN_PERIODS.gev_return_level_ci(
            100, 12.0, 1.5, -0.1, data_points=45
        )
        lower_180, upper_180 = RETURN_PERIODS.gev_return_level_ci(
            100, 12.0, 1.5, -0.1, data_points=180
        )

        self.assertAlmostEqual((upper_45 - lower_45) / (upper_180 - lower_180), 2.0)

    def test_interval_is_continuous_at_zero_shape(self):
        at_zero = RETURN_PERIODS.gev_return_level_ci(100, 12.0, 1.5, 0.0, 45)

        for shape in (-1e-4, 1e-4):
            nearby = RETURN_PERIODS.gev_return_level_ci(100, 12.0, 1.5, shape, 45)
            np.testing.assert_allclose(nearby, at_zero, atol=0.01)

    def test_no_interval_when_shape_is_half_or_more(self):
        lower, upper = RETURN_PERIODS.gev_return_level_ci(100, 12.0, 1.5, 0.5, 45)

        self.assertTrue(np.isnan(lower))
        self.assertTrue(np.isnan(upper))


class Phase4ExportTests(unittest.TestCase):
    def test_exports_native_and_3sec_gust_products(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            params_path = Path(temp_dir) / "gev_parameters.h5"
            latitude = np.array(
                [[35.0, 35.2], [36.1, 36.4]], dtype=np.float32
            )
            longitude = np.array(
                [[-100.0, -98.9], [-100.3, -98.5]], dtype=np.float32
            )
            with h5py.File(params_path, "w") as params:
                params.create_dataset("location", data=np.full((2, 2), 30.0))
                params.create_dataset("scale", data=np.full((2, 2), 5.0))
                params.create_dataset("shape", data=np.full((2, 2), -0.1))
                params.create_dataset("converged", data=np.ones((2, 2), dtype=bool))
                params.attrs["south_north"] = 2
                params.attrs["west_east"] = 2
                params.attrs["num_years"] = 45
                write_hdf5_coordinates(
                    params, latitude, longitude, "annual-test.nc"
                )

            with redirect_stdout(io.StringIO()):
                RETURN_PERIODS.main(temp_dir)

            netcdf_path = Path(temp_dir) / "return_periods" / "gev_return_periods.nc"
            with nc.Dataset(netcdf_path) as dataset:
                np.testing.assert_array_equal(
                    dataset.variables["return_period"][:],
                    [10, 25, 50, 100, 200, 500],
                )
                self.assertEqual(dataset.dimensions["return_period"].size, 6)
                self.assertEqual(dataset.gust_conversion_factor, 1.1176)
                self.assertEqual(dataset.source_duration_seconds, 20)
                self.assertEqual(dataset.target_duration_seconds, 3)
                self.assertEqual(dataset.Conventions, "CF-1.8")
                self.assertEqual(dataset.coordinate_grid, "curvilinear")
                self.assertIn(
                    "Log-linear interpolation",
                    dataset.durst_coefficient_20sec_derivation,
                )

                expected_variables = {
                    "wind_speed_native",
                    "wind_speed_native_lower_ci",
                    "wind_speed_native_upper_ci",
                    "wind_speed_3sec_gust",
                    "wind_speed_3sec_gust_lower_ci",
                    "wind_speed_3sec_gust_upper_ci",
                }
                self.assertTrue(expected_variables.issubset(dataset.variables))
                np.testing.assert_array_equal(dataset.variables["lat"][:], latitude)
                np.testing.assert_array_equal(dataset.variables["lon"][:], longitude)
                np.testing.assert_array_equal(dataset.variables["lat_idx"][:], [0, 1])
                np.testing.assert_array_equal(dataset.variables["lon_idx"][:], [0, 1])
                self.assertEqual(dataset.variables["lat"].dimensions, ("south_north", "west_east"))
                self.assertEqual(dataset.variables["lat"].units, "degrees_north")
                self.assertEqual(dataset.variables["lon"].units, "degrees_east")
                self.assertEqual(dataset.variables["crs"].grid_mapping_name, "latitude_longitude")
                self.assertEqual(dataset.variables["crs"].epsg_code, "EPSG:4326")
                self.assertIn(
                    'AUTHORITY["EPSG","4326"]',
                    dataset.variables["crs"].spatial_ref,
                )
                self.assertEqual(dataset.geospatial_bounds_crs, "EPSG:4326")

                for variable_name in expected_variables | {"rp_10_estimate"}:
                    self.assertEqual(
                        dataset.variables[variable_name].coordinates,
                        "lat_idx lon_idx lat lon",
                    )
                    self.assertEqual(dataset.variables[variable_name].grid_mapping, "crs")

                native = dataset.variables["wind_speed_native"][:]
                gust = dataset.variables["wind_speed_3sec_gust"][:]
                np.testing.assert_allclose(gust, native * 1.1176, rtol=1e-6)
                np.testing.assert_array_equal(
                    dataset.variables["rp_10_estimate"][:],
                    native[0],
                )

            csv_path = Path(temp_dir) / "return_periods" / "gev_return_periods.csv"
            csv_output = pd.read_csv(csv_path)
            self.assertIn("rp_10_3sec_gust", csv_output.columns)
            self.assertIn("rp_500_3sec_gust_upper", csv_output.columns)
            np.testing.assert_allclose(
                csv_output["latitude"].to_numpy(), latitude.ravel(), rtol=1e-7
            )
            np.testing.assert_allclose(
                csv_output["longitude"].to_numpy(), longitude.ravel(), rtol=1e-7
            )
            np.testing.assert_array_equal(
                csv_output["gust_conversion_factor"].to_numpy(),
                np.full(4, 1.1176),
            )


if __name__ == "__main__":
    unittest.main()
