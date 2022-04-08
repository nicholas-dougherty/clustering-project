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

###########
### Tried adding a box on top of a viz. 
#plt.figure(figsize=(16,8))
#sns.scatterplot(x='taxrate', y='log_abs', data=train, alpha=.4)
#plt.xlabel('Log Error |Abs Value|')
#plt.ylabel('Tax Rate')
#plt.title('Tax Rate Occasionally Factored into Outrageous Log Errors.')
#
### Creating the details to include in a box
##mu = train['log_abs'].mean()
##median = np.median(train['log_abs'])
##sigma = train['log_abs'].std()
##textstr = '\n'.join((
##    r'$\mu=%.2f$' % (mu, ),
##    r'$\mathrm{median}=%.2f$' % (median, ),
##    r'$\sigma=%.2f$' % (sigma, )))
##
### these are matplotlib.patch.Patch properties
##props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
##
### place a text box in upper left in axes coords
##ax.text(0.75, 0.9, textstr, transform=ax.transAxes, fontsize=10,
##        verticalalignment='top', bbox=props)
#plt.show()

#######

# from shapely.geometry import Point, Polygon
# import geopandas as gpd
# 
# geometry = [Point(xy) for xy in zip(train['longitude'], train['latitude'])]
# crs = {'init': 'epsg:4326'}
# geo_df = gpd.GeoDataFrame(train, crs = crs, geometry = geometry)
# m = folium_map
# 
# folium.Choropleth(geo_data= geo_df,
#     name='choropleth',
#     data=geo_df,
#     columns=['taxrate', 'age'],
#     key_on='feature.id',
#     fill_color='YlGn',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name='Tax Rate and Age %'
# ).add_to(m)

# was hoping to check bivariate relationships this way, but can't get it to work. 
# train = train.drop(columns='geometry')


miss_pct <- map_dbl(dtrain, function(x) { round((sum(is.na(x)) / length(x)) * 100, 1) })

miss_pct <- miss_pct[miss_pct > 0]

data.frame(miss=miss_pct, var=names(miss_pct), row.names=NULL) %>%
    ggplot(aes(x=reorder(var, -miss), y=miss)) + 
    geom_bar(stat='identity', fill='red') +
    labs(x='', y='% missing', title='Percent missing data by feature') +
    theme(axis.text.x=element_text(angle=90, hjust=1))
    
    #use in final report to show missing values