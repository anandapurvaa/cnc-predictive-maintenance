{{ config(materialized='view') }}

with raw_data as (
    select * from {{ source('raw_factory', 'cnc_telemetry_raw') }}
)

select
    -- Identifiers
    cast(machine_id as INT64) as machine_id,
    cast(product_id as STRING) as product_id,
    cast(type as STRING) as material_quality_type,

    -- Timestamps
    cast(timestamp_utc as TIMESTAMP) as reading_at,

    -- Sensor Metrics (Converted or Cleaned)
    cast(air_temperature_c as FLOAT64) as air_temperature_c,
    cast(process_temperature_c as FLOAT64) as process_temperature_c,
    cast(rotational_speed_rpm as INT64) as rotational_speed_rpm,
    cast(torque_nm as FLOAT64) as torque_nm,
    
    -- Handle the null values we injected into tool wear by defaulting them to 0 if missing
    coalesce(cast(tool_wear_min as INT64), 0) as tool_wear_min,

    -- Target Flag
    cast(failure_target as INT64) as has_failed

from raw_data