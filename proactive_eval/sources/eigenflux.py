"""Compatibility alias for historical EigenFlux experiment configurations."""

from proactive_eval.sources.alicerealm import (  # noqa: F401
    AliceRealmSource,
    sync_profiles,
)

EigenFluxSource = AliceRealmSource
