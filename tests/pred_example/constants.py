import numpy
lat_extract_start, lat_extract_end = 290, 290+128
lon_extract_start, lon_extract_end = 104, 104+128
zone = "NW"
coord = numpy.load('pred_example/radar_coords_NW.npz',allow_pickle=True)
longitudes = coord['lons'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]
latitudes = coord['lats'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]
del lat_extract_start, lat_extract_end, lon_extract_start, lon_extract_end, coord
