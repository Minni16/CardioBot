oex export
==========

Generated:        2026-05-09 15:02:34 UTC
oex version:      0.2.1
Project:          https://github.com/osgeonepal/oex

Country (ISO3):   NPL
Boundary:         geoBoundaries CGAZ ADM0
Bounding box:     (80.0601, 26.3474, 88.2043, 30.4731)

Dataset:          Hospitals
Format:           ESRI Shapefile (shp)
Features:         7,267

Source:           OpenStreetMap (Geofabrik NPL 2026-05-09)
Source URL:       https://www.openstreetmap.org/
Snapshot:         2026-05-09
License:          CDLA Permissive 2.0
License URL:      https://cdla.dev/permissive-2-0/

About the source
  OpenStreetMap is a community-edited geographic dataset of the world. Tag-
  based features (highway, building, amenity, ...) are extracted from the
  country PBF via quackosm.

Notes
  - Shapefile output is split by geometry type:
    <category>_polygons.shp, <category>_lines.shp, <category>_points.shp.
    This is a shapefile-format limitation, not a data limitation.
  - Field names are truncated to 10 characters in shp; gpkg keeps them full.

Feedback:         https://github.com/osgeonepal/oex/issues
Cache slug: hospitals