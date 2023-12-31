# model name - pretraining data - pretraining resolution for climaxv1
# model name - size - pretraining data - pretraining resolution
AVAILABLE_CMIP6_CKPTS = {
    'climaxv1-5.625': 'https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt',
    'climaxv1-1.40625': 'https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt',
    'climaxv2-small-1.40625': 'https://huggingface.co/tungnd/climax/resolve/main/vits.ckpt',
    'climaxv2-base-1.40625': 'https://huggingface.co/tungnd/climax/resolve/main/vitb.ckpt',
    'climaxv2-large-1.40625': 'https://huggingface.co/tungnd/climax/resolve/main/vitl.ckpt',
}

DEFAULT_VAR_CLIMAX_V1 = {
    "land_sea_mask",
    "orography",
    "lattitude",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential_50",
    "geopotential_250",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "u_component_of_wind_50",
    "u_component_of_wind_250",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "v_component_of_wind_50",
    "v_component_of_wind_250",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "temperature_50",
    "temperature_250",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "relative_humidity_50",
    "relative_humidity_250",
    "relative_humidity_500",
    "relative_humidity_600",
    "relative_humidity_700",
    "relative_humidity_850",
    "relative_humidity_925",
    "specific_humidity_50",
    "specific_humidity_250",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
}

DEFAULT_VAR_CLIMAX_V2 = [
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "vertical_velocity_50",
    "vertical_velocity_100",
    "vertical_velocity_150",
    "vertical_velocity_200",
    "vertical_velocity_250",
    "vertical_velocity_300",
    "vertical_velocity_400",
    "vertical_velocity_500",
    "vertical_velocity_600",
    "vertical_velocity_700",
    "vertical_velocity_850",
    "vertical_velocity_925",
    "vertical_velocity_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]