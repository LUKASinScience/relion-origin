"""Class-average thumbnails for the GUI -- reuses reliontree's own MRC
reader (pure stdlib: struct + zlib, no new dependency, no numpy/Pillow)
instead of adding an MRC library just to grab a middle slice for display.

Answers a concrete need: a QC candidate list of particle IDs means little
without seeing what the class average they're being compared against
actually looks like.
"""

from pathlib import Path

try:
    from reliontree.relion_artifacts import list_class_maps
    from reliontree.relion_mrc_thumb import mrc_slice_thumbnail
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "relion_origin.class_images needs the reliontree package (pip install reliontree) "
        "for MRC thumbnail rendering"
    ) from _exc


def class_average_thumbnails(job_dir, size=160):
    """{class_number: data-URI PNG} for every per-class average volume in
    a Class2D/Class3D/Refine3D job directory. RELION numbers classes 1..K
    contiguously, matching `list_class_maps`'s sorted order -- skips a
    class whose MRC fails to decode (unsupported mode, truncated file)
    rather than breaking the whole set.
    """
    thumbnails = {}
    for class_number, path in enumerate(list_class_maps(Path(job_dir)), start=1):
        uri = mrc_slice_thumbnail(path, size=size)
        if uri:
            thumbnails[class_number] = uri
    return thumbnails
