"""NeuroEmbed API — hosted REVE EEG foundation-model inference.

The package is intentionally import-light at top level. The REVE model
itself is loaded lazily in neuroembed.core.reve so that lightweight API
processes (health checks, auth) can start without a GPU.
"""

__version__ = "0.1.0"
