{{ config(materialized='table') }}

with staging_telemetry as (
    select * from {{ ref('stg_cnc_telemetry') }}
)

select
    -- Primary Keys / Identifiers
    generate_uuid() as telemetry_fact_id,
    machine_id,
    product_id,
    material_quality_type,
    
    -- Telemetry Details
    reading_at,
    air_temperature_c,
    process_temperature_c,
    
    -- Operational Deltas (Calculating temperature differentials)
    round(process_temperature_c - air_temperature_c, 2) as temperature_differential_c,
    
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,

    -- Risk Assessment Metric Flags
    case 
        when rotational_speed_rpm > 2500 then 1 
        else 0 
    end as is_high_speed_anomalous,

    case 
        when tool_wear_min >= 200 then 1 
        else 0 
    end as is_critical_wear_risk,

    -- Ultimate Target
    has_failed

from staging_telemetry