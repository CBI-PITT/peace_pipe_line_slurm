from analysis.utils import ensure_tiffs_extracted


def extract_tiff_series(options):
    for channel in range(options['channels']):
        ensure_tiffs_extracted(channel, options)
