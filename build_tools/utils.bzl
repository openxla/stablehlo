"""A collection of starlark functions for use in BUILD files that are useful for StableHLO."""

def is_bzlmod_enabled():
    """Determine whether bzlmod mode is enabled."""

    # If bzlmod is enabled, then `str(Label(...))` returns a canonical label,
    # these start with `@@`.
    return str(Label("//:invalid")).startswith("@@")

def workspace_name():
    """Return the name of the workspace."""
    return "_main" if is_bzlmod_enabled() else "stablehlo"
