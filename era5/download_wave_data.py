import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'eastward_turbulent_surface_stress', 'instantaneous_eastward_turbulent_surface_stress', 'instantaneous_northward_turbulent_surface_stress',
            'mean_wave_direction_of_first_swell_partition', 'mean_wave_direction_of_second_swell_partition', 'mean_wave_direction_of_third_swell_partition',
            'mean_wave_period_of_first_swell_partition', 'mean_wave_period_of_second_swell_partition', 'mean_wave_period_of_third_swell_partition',
            'northward_turbulent_surface_stress', 'significant_wave_height_of_first_swell_partition', 'significant_wave_height_of_second_swell_partition',
            'significant_wave_height_of_third_swell_partition',
        ],
        'year': '2019',
        'month': '10',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    },
    'download.nc')
