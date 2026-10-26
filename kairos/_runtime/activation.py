"""Private compatibility activation for the frozen serving runtime.

The public experiment surface is intentionally limited to Kairos and
Continuum. The pinned engine still uses internal environment flags and an
activation descriptor; keep that translation here rather than exposing
implementation-era labels in the public configuration files.
"""

RUNTIME_ACTIVATION = {
    'kairos': 'v21-eager-a~r5callcommitment~progressivewindow~jointmean~localeta~provenanceprior~onlinefitresidual~knownoutput~finishlogical~kairos',
    'continuum': 'continuum-index-partial-eager-a',
}

RUNTIME_ENVIRONMENT = {
    'kairos': {
        'CONTINUUM_R5': '1',
        'CONTINUUM_R82': '1',
        'CONTINUUM_R82_FOREST': '${ROOT}/assets/forest.json',
        'CONTINUUM_R82_PROFILE': '${ROOT}/assets/reference_profile.json',
        'CONTINUUM_R83': '1',
        'CONTINUUM_R83_PRELOAD': '1',
    },
    'continuum': {},
}
