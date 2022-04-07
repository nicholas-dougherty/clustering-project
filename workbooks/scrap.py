#cali = folium.Map(location=[34.0522, -118.2437],
#                  tiles='cartodbpositron'
#                 )
#train2.apply(lambda row: folium.CircleMarker(location=[row["latitude"], row["longitude"]]).add_to(cali), axis=1)

###Used to move decimals originally. Code for moving in wrangle. 
# parts = train['latitude'].copy()
# parts
#carts = train['longitude'].copy()
#carts 
# phew = moveDecimalPoint(parts, -6)
# chew = moveDecimalPoint(carts, -6)

##### tinking with geopandas
#street_map = gpd.read_file('/Users/nicholasdougherty/Downloads/tl_2018_06037_roads/tl_2018_06037_roads.shp')

#fig, ax = plt.subplots(figsize = (15,15))
#street_map.plot(ax = ax)

#from shapely.geometry import Point, Polygon
#crs = {'init': 'epsg:4326'}

## latitude and longitude information can be used to create Points.
## a point is essentially a single object descirbing both as a data-point.
## specify the longitude before latitude column

#geometry = [Point(xy) for xy in zip(train2['longitude'], #train2['latitude'])]
#geometry[:3]

## create a GeoDataFrame
#geo_df = gpd.GeoDataFrame(train2, crs = crs, geometry = geometry)
#geo_df.head()

#fig, ax = plt.subplots(figsize = (15, 15))
#street_map.plot(ax = ax, alpha = 0.4, color='grey')
##geo_df[geo_df['county' == 'los_angeles']].plot(ax = ax, markersize = 20)
#geo_df.plot(ax = ax, markersize = 20)

###########
## still may try this, but with taxes rather than expenses
# # log transformation
# df('log expenses' ] = np. log2 (df('expenses' ] +1)
# plt.figure(1)
# df('expenses' ].plot (kind = 'hist')
# plt.figure(2)
# df('log expenses' ].plot (kind = 'hist')