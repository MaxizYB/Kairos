"""Translate the public Kairos/Continuum choices to native lifecycle settings."""

RUNTIME_ACTIVATION = {
    "kairos": "continuity-eager-a~provenanceprior~onlinefitresidual~finishlogical~kairos",
    "continuum": "continuum-index-partial-eager-a",
}

RUNTIME_ENVIRONMENT = {
    "kairos": {
        "CONTINUUM_HOLDING": "1",
        "KAIROS_PREDICTOR_FOREST": "${ROOT}/assets/forest.json",
        "KAIROS_PREDICTOR_PROFILE": "${ROOT}/assets/reference_profile.json",
        "KAIROS_PREDICTION_PRELOAD": "1",
        "CONTINUUM_CURRENT_OUTPUT_FOREST": "${ROOT}/assets/current_output.json",
        "CONTINUUM_CURRENT_OUTPUT_FALLBACK": "${ROOT}/assets/current_output_metadata.json",
        "KAIROS_FORECAST_BACKEND": "1",
        "KAIROS_STARVATION_GUARD": "1",
        "KAIROS_STARVATION_THRESHOLD_S": "4.0",
        "KAIROS_STARVATION_QUANTUM_S": "1.0",
    },
    "continuum": {},
}
