"""Run brainreg with the BrainGlobe remote version check disabled.

bg_atlasapi fetches last_versions.conf from gin.g-node.org on every
BrainGlobeAtlas init; on the compute nodes the proxy answers with a 403
HTML page, which configparser chokes on (MissingSectionHeaderError).
The version check is irrelevant for PEACE runs, so skip it and use the
locally installed atlas.
"""
import sys

import bg_atlasapi.bg_atlas

bg_atlasapi.bg_atlas.BrainGlobeAtlas.check_latest_version = lambda self: None

from brainreg.cli import main

if __name__ == "__main__":
    sys.argv[0] = "brainreg"
    main()
