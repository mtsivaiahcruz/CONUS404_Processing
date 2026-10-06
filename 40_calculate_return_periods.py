"""
Phases 3 & 4: Calculate return period wind speeds with confidence intervals and export results.

This script:
1. Loads GEV parameters (μ, σ, ξ) from Phase 2
2. Calculates return level for each return period using GEV inverse CDF
3. Computes 95% confidence intervals via delta method
4. Converts native 20-second return levels to 3-second engineering gusts
5. Exports native and converted results to NetCDF and CSV formats
6. Generates quality report
"""

import os
import h5py
import numpy as np
import netCDF4 as nc
import pandas as pd
from scipy.special import digamma, gamma
from scipy.stats import genextreme, norm
from datetime import datetime

from coordinate_utils import read_hdf5_coordinates

WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
    'AUTHORITY["EPSG","4326"]]'
)

def convert_20s_to_3s_gust(v_base_20s):
    """Convert CONUS404 20-second wind maxima to 3-second gust speeds.

    The Durst Curve relates maximum wind speeds averaged over a specified
    duration to an hourly mean wind speed. This conversion uses the ratio of
    the ASCE 7 Durst coefficients for 3 seconds (1.52) and 20 seconds (1.36)
    to align CONUS404 ``wrfxtrm`` wind maxima, interpreted at the 20-second
    model time step, with the 3-second gust duration used by engineering
    applications such as Hazus fragility functions. The 20-second coefficient
    is derived by log-linear interpolation of the 10- and 30-second empirical
    Durst values. The coefficients assume the standard open-terrain exposure
    convention.

    Args:
        v_base_20s: Scalar or NumPy array of native 20-second wind speeds.

    Returns:
        The input wind speeds multiplied by the rounded Durst coefficient
        ratio. Array shapes and NaN values are preserved. This deterministic
        conversion does not add uncertainty beyond that of the native values.
    """
    DURST_3S = 1.52
    DURST_20S = 1.36
    GUST_FACTOR = round(DURST_3S / DURST_20S, 4)

    return v_base_20s * GUST_FACTOR

def gev_return_level(return_period, location, scale, shape):
    """
    Calculate GEV return level for a given return period.
    
    Formula: x = μ + (σ/ξ) × [1 - (-log(1 - 1/R))^ξ]
    As ξ → 0 this becomes the Gumbel formula: x = μ - σ × log(-log(1 - 1/R))
    
    Args:
        return_period: Return period in years (R)
        location: GEV location parameter (μ)
        scale: GEV scale parameter (σ)
        shape: GEV shape parameter (ξ)
    
    Returns:
        Return level (wind speed)
    """
    if scale <= 0 or np.isnan(location) or np.isnan(scale) or np.isnan(shape):
        return np.nan
    
    try:
        p = 1.0 - 1.0 / return_period
        
        if np.abs(shape) < 1e-6:  # Gumbel case (ξ ≈ 0)
            return location - scale * np.log(-np.log(p))
        else:
            return location + (scale / shape) * (1.0 - (-np.log(p)) ** shape)
    except:
        return np.nan

def gev_information_matrix(shape):
    """
    Expected Fisher information for one observation from a GEV distribution
    with location 0 and scale 1 (Prescott and Walden, 1980, Biometrika 67).
    
    Args:
        shape: GEV shape parameter (ξ), which must be below 0.5
    
    Returns:
        3 × 3 array with rows and columns ordered (location, scale, shape)
    """
    # The formulas below divide by the shape up to four times, so they lose
    # precision close to zero. Inside ±0.005, interpolate between the two edges.
    near_zero = 0.005
    if abs(shape) < near_zero:
        below = gev_information_matrix(-near_zero)
        above = gev_information_matrix(near_zero)
        weight = (shape + near_zero) / (2.0 * near_zero)
        return (1.0 - weight) * below + weight * above
    
    k = shape
    euler = np.euler_gamma
    g = gamma(2.0 - k)
    p = (1.0 - k) ** 2 * gamma(1.0 - 2.0 * k)
    q = g * (digamma(1.0 - k) - (1.0 - k) / k)
    
    loc_loc = p
    loc_scale = (p - g) / k
    loc_shape = -(q + p / k) / k
    scale_scale = (1.0 - 2.0 * g + p) / k**2
    scale_shape = (1.0 - euler - (1.0 - g) / k - q - p / k) / k**2
    shape_shape = (np.pi**2 / 6.0 + (1.0 - euler - 1.0 / k) ** 2 + 2.0 * q / k + p / k**2) / k**2
    
    return np.array([
        [loc_loc, loc_scale, loc_shape],
        [loc_scale, scale_scale, scale_shape],
        [loc_shape, scale_shape, shape_shape]
    ])

def gev_parameter_covariance(scale, shape, data_points):
    """
    Covariance of the fitted GEV parameters for a record of `data_points` years.
    
    This is the inverse of the expected Fisher information. It is the standard
    large-sample covariance of a maximum likelihood fit.
    
    Returns:
        3 × 3 array with rows and columns ordered (location, scale, shape)
    """
    unit_covariance = np.linalg.inv(gev_information_matrix(shape))
    
    # Location and scale are wind speeds, so their rows and columns scale with σ
    units = np.array([scale, scale, 1.0])
    return unit_covariance * np.outer(units, units) / data_points

def gev_return_level_gradient(return_period, scale, shape):
    """
    Derivatives of the return level with respect to (location, scale, shape).
    """
    y = -np.log(1.0 - 1.0 / return_period)
    log_y = np.log(y)
    
    if abs(shape) > 1e-5:
        d_scale = (1.0 - y**shape) / shape
        d_shape = -scale * ((1.0 - y**shape) / shape**2 + y**shape * log_y / shape)
    else:
        # Gumbel limit (ξ -> 0)
        d_scale = -log_y
        d_shape = -0.5 * scale * log_y**2
    
    return np.array([1.0, d_scale, d_shape])

def gev_return_level_ci(return_period, location, scale, shape, data_points=40):
    """
    Calculate the 95% confidence interval for a GEV return level using the delta method.
    
    The variance of the return level is g' V g, where g holds the derivatives of
    the return level with respect to the three parameters and V is the covariance
    of the fitted parameters.
    
    V assumes a maximum likelihood fit. For parameters fitted another way, such
    as L-moments, the interval is an approximation.
    
    Args:
        return_period: Return period in years (R)
        location: GEV location parameter (μ)
        scale: GEV scale parameter (σ)
        shape: GEV shape parameter (ξ)
        data_points: Number of annual maxima the parameters were fitted to
    
    Returns:
        (lower, upper) bounds. Both are NaN when shape >= 0.5, where the
        covariance does not exist.
    """
    # Calculate return level
    rp_level = gev_return_level(return_period, location, scale, shape)
    
    if np.isnan(rp_level) or shape >= 0.5:
        return np.nan, np.nan
    
    try:
        gradient = gev_return_level_gradient(return_period, scale, shape)
        covariance = gev_parameter_covariance(scale, shape, data_points)
        
        # Standard error estimate
        se = np.sqrt(gradient @ covariance @ gradient)
        
        # 95% CI (z = 1.96)
        ci_width = 1.96 * se
        lower = rp_level - ci_width
        upper = rp_level + ci_width
        
        return lower, upper
    except:
        return np.nan, np.nan

def main(base_dir=None):
    # Paths
    if base_dir is None:
        base_dir = '/Users/andrewjohnson/Documents/GitHub/CONUS404_Processing/output'
    params_file = os.path.join(base_dir, 'gev_parameters.h5')
    output_dir = os.path.join(base_dir, 'return_periods')
    output_nc = os.path.join(output_dir, 'gev_return_periods.nc')
    output_csv = os.path.join(output_dir, 'gev_return_periods.csv')
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Return periods to calculate
    return_periods = [10, 25, 50, 100, 200, 500]
    
    print("=" * 80)
    print("PHASES 3 & 4: RETURN PERIOD CALCULATION AND EXPORT")
    print("=" * 80)
    
    # Check input file
    if not os.path.exists(params_file):
        print(f"Error: Input file not found: {params_file}")
        print("Please run fit_gev_parameters.py first")
        return
    
    # Load GEV parameters
    print(f"\nLoading GEV parameters from {params_file}...")
    try:
        with h5py.File(params_file, 'r') as f:
            location_grid = f['location'][:]
            scale_grid = f['scale'][:]
            shape_grid = f['shape'][:]
            converged_grid = f['converged'][:]
            
            south_north = f.attrs['south_north']
            west_east = f.attrs['west_east']
            num_years = f.attrs['num_years']
            latitude, longitude = read_hdf5_coordinates(
                f, (south_north, west_east)
            )
            coordinate_source = f.attrs.get('coordinate_source', params_file)
    except Exception as e:
        print(f"Error loading HDF5: {str(e)}")
        return
    
    print(f"  Grid: {south_north} × {west_east} = {south_north * west_east:,} grid points")
    print(f"  Converged fits: {np.sum(converged_grid)} / {south_north * west_east:,}")
    
    print(f"\nCalculating return levels for: {return_periods}")
    
    # Initialize result grids (return period, 3 columns: estimate, lower, upper)
    result_grids = {}
    for rp in return_periods:
        result_grids[rp] = {
            'estimate': np.full((south_north, west_east), np.nan, dtype=np.float32),
            'lower': np.full((south_north, west_east), np.nan, dtype=np.float32),
            'upper': np.full((south_north, west_east), np.nan, dtype=np.float32)
        }
    
    # Calculate return levels for all grid points
    total_points = south_north * west_east
    for lat in range(south_north):
        for lon in range(west_east):
            # Only calculate if fit converged
            if converged_grid[lat, lon]:
                loc = location_grid[lat, lon]
                scale = scale_grid[lat, lon]
                shp = shape_grid[lat, lon]
                
                for rp in return_periods:
                    estimate = gev_return_level(rp, loc, scale, shp)
                    lower, upper = gev_return_level_ci(rp, loc, scale, shp, data_points=num_years)
                    
                    result_grids[rp]['estimate'][lat, lon] = estimate
                    result_grids[rp]['lower'][lat, lon] = lower
                    result_grids[rp]['upper'][lat, lon] = upper
        
        # Progress update
        if (lat + 1) % 100 == 0:
            pct = 100 * (lat + 1) / south_north
            print(f"  Progress: {lat + 1} / {south_north} rows ({pct:.1f}%)")
    
    print("✓ Return level calculation complete")

    # Apply the duration conversion after GEV fitting and return-level calculation.
    gust_factor = convert_20s_to_3s_gust(1.0)
    gust_result_grids = {}
    for rp in return_periods:
        gust_result_grids[rp] = {
            'estimate': convert_20s_to_3s_gust(result_grids[rp]['estimate']),
            'lower': convert_20s_to_3s_gust(result_grids[rp]['lower']),
            'upper': convert_20s_to_3s_gust(result_grids[rp]['upper'])
        }

    print(f"✓ Applied Durst 20-second to 3-second gust factor: {gust_factor:.4f}")
    
    # Create NetCDF output
    print(f"\nCreating NetCDF output: {output_nc}...")
    try:
        ds_out = nc.Dataset(output_nc, 'w', format='NETCDF4')
        
        # Dimensions
        ds_out.createDimension('return_period', len(return_periods))
        ds_out.createDimension('south_north', south_north)
        ds_out.createDimension('west_east', west_east)
        
        # Global attributes
        ds_out.title = 'CONUS404 GEV Return Period Analysis'
        ds_out.description = 'Native and Durst-converted Generalized Extreme Value return periods for SPDUV10MAX'
        ds_out.variable = 'SPDUV10MAX (Daily maximum wind speed at 10 meters)'
        ds_out.units = 'm/s'
        ds_out.method = 'Block Maxima (annual) fitted to GEV via MLE'
        ds_out.num_years = num_years
        ds_out.confidence_level = '95%'
        ds_out.ci_method = 'Delta method with the expected Fisher information of the GEV'
        ds_out.created_date = datetime.now().isoformat()
        ds_out.grid_dimensions = f'{south_north} x {west_east}'
        ds_out.Conventions = 'CF-1.8'
        ds_out.coordinate_grid = 'curvilinear'
        ds_out.coordinate_source = coordinate_source
        ds_out.gust_conversion_factor = gust_factor
        ds_out.durst_coefficient_3sec = 1.52
        ds_out.durst_coefficient_20sec = 1.36
        ds_out.durst_coefficient_20sec_derivation = 'Log-linear interpolation of the 10-second and 30-second empirical Durst values'
        ds_out.source_duration_seconds = 20
        ds_out.target_duration_seconds = 3
        ds_out.source_duration_interpretation = 'CONUS404 wrfxtrm SPDUV10MAX maxima at the 20-second model time step'
        ds_out.gust_conversion_method = 'Durst Curve coefficient ratio: C_Durst(3 s) / C_Durst(20 s)'
        ds_out.gust_conversion_reference = 'ASCE 7 Commentary Figure C26.5-1, Maximum Speed Averaged over t to Hourly Mean Speed'
        ds_out.exposure_convention = 'Standard open terrain'
        ds_out.gust_conversion_uncertainty = 'No additional conversion uncertainty modeled; native estimates and CI bounds are multiplied by the same deterministic factor'

        rp_variable = ds_out.createVariable('return_period', 'i4', ('return_period',))
        rp_variable.long_name = 'Return period'
        rp_variable.units = 'years'
        rp_variable[:] = return_periods

        spatial_dimensions = ('south_north', 'west_east')
        latitude_variable = ds_out.createVariable(
            'lat', 'f4', spatial_dimensions, zlib=True, complevel=4
        )
        latitude_variable.standard_name = 'latitude'
        latitude_variable.long_name = 'CONUS404 grid-cell latitude'
        latitude_variable.units = 'degrees_north'
        latitude_variable[:] = latitude

        longitude_variable = ds_out.createVariable(
            'lon', 'f4', spatial_dimensions, zlib=True, complevel=4
        )
        longitude_variable.standard_name = 'longitude'
        longitude_variable.long_name = 'CONUS404 grid-cell longitude'
        longitude_variable.units = 'degrees_east'
        longitude_variable[:] = longitude

        output_dimensions = ('return_period', 'south_north', 'west_east')
        dimensioned_variables = {
            'wind_speed_native': ds_out.createVariable('wind_speed_native', 'f4', output_dimensions),
            'wind_speed_native_lower_ci': ds_out.createVariable('wind_speed_native_lower_ci', 'f4', output_dimensions),
            'wind_speed_native_upper_ci': ds_out.createVariable('wind_speed_native_upper_ci', 'f4', output_dimensions),
            'wind_speed_3sec_gust': ds_out.createVariable('wind_speed_3sec_gust', 'f4', output_dimensions),
            'wind_speed_3sec_gust_lower_ci': ds_out.createVariable('wind_speed_3sec_gust_lower_ci', 'f4', output_dimensions),
            'wind_speed_3sec_gust_upper_ci': ds_out.createVariable('wind_speed_3sec_gust_upper_ci', 'f4', output_dimensions)
        }

        dimensioned_variables['wind_speed_native'].long_name = 'Native CONUS404 20-second model-time-step return-level estimate'
        dimensioned_variables['wind_speed_native_lower_ci'].long_name = 'Native CONUS404 return-level 95% CI lower bound'
        dimensioned_variables['wind_speed_native_upper_ci'].long_name = 'Native CONUS404 return-level 95% CI upper bound'
        dimensioned_variables['wind_speed_3sec_gust'].long_name = 'Durst-converted 3-second gust return-level estimate'
        dimensioned_variables['wind_speed_3sec_gust_lower_ci'].long_name = 'Durst-converted 3-second gust return-level 95% CI lower bound'
        dimensioned_variables['wind_speed_3sec_gust_upper_ci'].long_name = 'Durst-converted 3-second gust return-level 95% CI upper bound'

        for variable in dimensioned_variables.values():
            variable.units = 'm/s'
            variable.coordinates = 'lat lon'

        for variable_name in (
            'wind_speed_3sec_gust',
            'wind_speed_3sec_gust_lower_ci',
            'wind_speed_3sec_gust_upper_ci'
        ):
            dimensioned_variables[variable_name].gust_conversion_factor = gust_factor
            dimensioned_variables[variable_name].source_duration_seconds = 20
            dimensioned_variables[variable_name].target_duration_seconds = 3
            dimensioned_variables[variable_name].gust_conversion_method = 'Durst Curve coefficient ratio'

        for rp_index, rp in enumerate(return_periods):
            dimensioned_variables['wind_speed_native'][rp_index, :, :] = result_grids[rp]['estimate']
            dimensioned_variables['wind_speed_native_lower_ci'][rp_index, :, :] = result_grids[rp]['lower']
            dimensioned_variables['wind_speed_native_upper_ci'][rp_index, :, :] = result_grids[rp]['upper']
            dimensioned_variables['wind_speed_3sec_gust'][rp_index, :, :] = gust_result_grids[rp]['estimate']
            dimensioned_variables['wind_speed_3sec_gust_lower_ci'][rp_index, :, :] = gust_result_grids[rp]['lower']
            dimensioned_variables['wind_speed_3sec_gust_upper_ci'][rp_index, :, :] = gust_result_grids[rp]['upper']
        
        # Retain native per-period variables for downstream compatibility.
        for rp in return_periods:
            var_estimate = ds_out.createVariable(f'rp_{rp}_estimate', 'f4', ('south_north', 'west_east'))
            var_lower = ds_out.createVariable(f'rp_{rp}_lower_ci', 'f4', ('south_north', 'west_east'))
            var_upper = ds_out.createVariable(f'rp_{rp}_upper_ci', 'f4', ('south_north', 'west_east'))
            
            var_estimate.long_name = f'{rp}-year return level (point estimate)'
            var_lower.long_name = f'{rp}-year return level (95% CI lower bound)'
            var_upper.long_name = f'{rp}-year return level (95% CI upper bound)'
            
            var_estimate.units = 'm/s'
            var_lower.units = 'm/s'
            var_upper.units = 'm/s'
            var_estimate.coordinates = 'lat lon'
            var_lower.coordinates = 'lat lon'
            var_upper.coordinates = 'lat lon'
            var_estimate.compatibility_alias_for = f'wind_speed_native at return_period={rp}'
            var_lower.compatibility_alias_for = f'wind_speed_native_lower_ci at return_period={rp}'
            var_upper.compatibility_alias_for = f'wind_speed_native_upper_ci at return_period={rp}'
            
            var_estimate[:] = result_grids[rp]['estimate']
            var_lower[:] = result_grids[rp]['lower']
            var_upper[:] = result_grids[rp]['upper']
        
        ds_out.close()
        print(f"✓ NetCDF saved: {output_nc}")
        print(f"  File size: {os.path.getsize(output_nc) / 1e6:.2f} MB")
    
    except Exception as e:
        print(f"✗ Error creating NetCDF: {str(e)}")
        return
    
    # Create CSV output
    print(f"\nCreating CSV output: {output_csv}...")
    try:
        csv_data = {
            'lat_idx': [],
            'lon_idx': [],
            'latitude': [],
            'longitude': [],
        }
        
        # Add columns for each return period
        for rp in return_periods:
            csv_data[f'rp_{rp}'] = []
            csv_data[f'rp_{rp}_lower'] = []
            csv_data[f'rp_{rp}_upper'] = []
            csv_data[f'rp_{rp}_3sec_gust'] = []
            csv_data[f'rp_{rp}_3sec_gust_lower'] = []
            csv_data[f'rp_{rp}_3sec_gust_upper'] = []
        
        csv_data['gust_conversion_factor'] = []
        csv_data['converged'] = []
        
        # Populate data
        for lat in range(south_north):
            for lon in range(west_east):
                csv_data['lat_idx'].append(lat)
                csv_data['lon_idx'].append(lon)
                csv_data['latitude'].append(latitude[lat, lon])
                csv_data['longitude'].append(longitude[lat, lon])
                
                for rp in return_periods:
                    csv_data[f'rp_{rp}'].append(result_grids[rp]['estimate'][lat, lon])
                    csv_data[f'rp_{rp}_lower'].append(result_grids[rp]['lower'][lat, lon])
                    csv_data[f'rp_{rp}_upper'].append(result_grids[rp]['upper'][lat, lon])
                    csv_data[f'rp_{rp}_3sec_gust'].append(gust_result_grids[rp]['estimate'][lat, lon])
                    csv_data[f'rp_{rp}_3sec_gust_lower'].append(gust_result_grids[rp]['lower'][lat, lon])
                    csv_data[f'rp_{rp}_3sec_gust_upper'].append(gust_result_grids[rp]['upper'][lat, lon])
                
                csv_data['gust_conversion_factor'].append(gust_factor)
                csv_data['converged'].append(int(converged_grid[lat, lon]))
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv, index=False)
        
        print(f"✓ CSV saved: {output_csv}")
        print(f"  Rows: {len(df):,} (grid points)")
        print(f"  Columns: {len(df.columns)}")
        print(f"  File size: {os.path.getsize(output_csv) / 1e6:.2f} MB")
    
    except Exception as e:
        print(f"✗ Error creating CSV: {str(e)}")
        return
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("RETURN PERIOD STATISTICS")
    print("=" * 80)
    
    for rp in return_periods:
        estimates = result_grids[rp]['estimate'][~np.isnan(result_grids[rp]['estimate'])]
        lowers = result_grids[rp]['lower'][~np.isnan(result_grids[rp]['lower'])]
        uppers = result_grids[rp]['upper'][~np.isnan(result_grids[rp]['upper'])]
        
        if len(estimates) > 0:
            gust_estimates = gust_result_grids[rp]['estimate'][~np.isnan(gust_result_grids[rp]['estimate'])]
            print(f"\n{rp}-year Return Period:")
            print(f"  Native estimate: {np.mean(estimates):.2f} ± {np.std(estimates):.2f} m/s")
            print(f"  Native range: {np.min(estimates):.2f} - {np.max(estimates):.2f} m/s")
            print(f"  Native CI width (mean): {np.mean(uppers - lowers):.2f} m/s")
            print(f"  3-second gust estimate: {np.mean(gust_estimates):.2f} ± {np.std(gust_estimates):.2f} m/s")
    
    # Monotonicity check
    print("\n" + "=" * 80)
    print("VERIFICATION: MONOTONIC INCREASE")
    print("=" * 80)
    
    monotonic = True
    for lat in range(south_north):
        for lon in range(west_east):
            estimates = [result_grids[rp]['estimate'][lat, lon] for rp in return_periods]
            valid_estimates = [e for e in estimates if not np.isnan(e)]
            
            if len(valid_estimates) > 1:
                if not all(valid_estimates[i] <= valid_estimates[i+1] for i in range(len(valid_estimates)-1)):
                    monotonic = False
                    break
    
    if monotonic:
        print("✓ All return levels increase monotonically (as expected)")
    else:
        print("✗ WARNING: Some return levels do not increase monotonically!")
        print("  This may indicate fitting issues at certain grid points")

    ratio_consistent = True
    for rp in return_periods:
        for result_name in ('estimate', 'lower', 'upper'):
            expected = result_grids[rp][result_name] * gust_factor
            if not np.allclose(
                gust_result_grids[rp][result_name],
                expected,
                rtol=0.0,
                atol=0.0,
                equal_nan=True
            ):
                ratio_consistent = False
                break

    if ratio_consistent:
        print(f"✓ All 3-second gust values equal native values × {gust_factor:.4f}")
    else:
        print("✗ WARNING: Some 3-second gust values do not match the Durst conversion factor")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {output_nc} (NetCDF - for GIS/spatial analysis)")
    print(f"  2. {output_csv} (CSV - for tabular analysis)")
    print(f"\nYou can now:")
    print(f"  - Use the NetCDF file in ArcGIS, QGIS, or Python for mapping")
    print(f"  - Import the CSV into Excel, R, or statistical software")
    print(f"  - Query specific lat/lon return periods from either format")

if __name__ == '__main__':
    main()
