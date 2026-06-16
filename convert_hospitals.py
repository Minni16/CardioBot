import shapefile
import json
import os

BASE = 'Machine_Learning/osgeonepal_npl_hospitals_osm_shp'
OUTPUT = 'Machine_Learning/hospitals_nepal.json'

INCLUDE_AMENITIES = {'hospital', 'clinic', 'doctors'}
EXCLUDE_NAMES = ['veterinary', 'animal', 'dental', 'dentist', 'optical', 'pharmacy', 'eye bank']

def best_name(rec):
    for field in ('name_latin', 'name_en', 'name'):
        val = rec.get(field, '') or ''
        val = val.strip()
        if val:
            return val
    return None

def is_relevant(name):
    n = name.lower()
    return not any(x in n for x in EXCLUDE_NAMES)

def centroid(points):
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return sum(lats) / len(lats), sum(lons) / len(lons)

hospitals = []

for shp_file, geom_type in [
    (os.path.join(BASE, 'hospitals_points.shp'), 'point'),
    (os.path.join(BASE, 'hospitals_polygons.shp'), 'polygon'),
]:
    sf = shapefile.Reader(shp_file, encoding='utf-8')
    fields = [f[0] for f in sf.fields[1:]]
    count = 0
    for shape_rec in sf.iterShapeRecords():
        rec = dict(zip(fields, shape_rec.record))
        amenity = (rec.get('amenity') or '').strip().lower()
        if amenity not in INCLUDE_AMENITIES:
            continue
        name = best_name(rec)
        if not name or not is_relevant(name):
            continue
        shape = shape_rec.shape
        if geom_type == 'point':
            if not shape.points:
                continue
            lon, lat = shape.points[0]
        else:
            if not shape.points:
                continue
            lat, lon = centroid(shape.points)
        hospitals.append({
            'name': name,
            'type': amenity,
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'district': (rec.get('adm2_name') or '').strip(),
            'municipality': (rec.get('adm3_name') or '').strip(),
        })
        count += 1
    print(f'{shp_file}: {count} records added')

print(f'Total: {len(hospitals)} hospitals/clinics')
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(hospitals, f, ensure_ascii=False, separators=(',', ':'))
print(f'Saved to {OUTPUT}')
