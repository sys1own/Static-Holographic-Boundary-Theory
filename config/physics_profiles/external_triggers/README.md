Drop external trigger files here for `shbt.observational_bridge`.

Supported formats:
- `*.json`: a single trigger object, a list of trigger objects, or a top-level object with `triggers`, `events`, `observations`, or `records`
- `*.csv`: header row plus one trigger per row

Remote sources can be mirrored into this directory with
`shbt.http_bridge.stage_trigger_payload(...)` or the convenience wrappers on
`shbt.observational_bridge.ObservationalBridge`.

Common fields:
- `source` / `instrument`
- `event_name` / `event_id`
- `redshift` (or `z`)
- `observed_expansion_rate_km_s_mpc` for direct H(z) comparisons

Optional LIGO fields:
- `luminosity_distance_mpc`
- `peak_strain`

Optional JWST fields:
- `galaxy_count`
- `reference_galaxy_count`
- `reference_expansion_rate_km_s_mpc`

If a JWST trigger omits `observed_expansion_rate_km_s_mpc`, the bridge can rescale the branch prediction by `reference_galaxy_count / galaxy_count` when both counts are present.
