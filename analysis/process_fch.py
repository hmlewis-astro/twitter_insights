import pandas as pd

def process(thandle):
	print('Formatting historical follower count archive...')
	archive_init = pd.read_csv('{}/follower_count.csv'.format(thandle))
	
	archive_init = archive_init.drop_duplicates(subset=['MementoDatetime']).reset_index(drop=True)
	archive_init = archive_init.drop_duplicates(subset=['FollowerCount', 'AbsGrowth']).reset_index(drop=True)

	archive = archive_init.copy()
	archive = archive.drop(['AbsGrowth', 'RelGrowth', 'AbsPerGrowth', 'RelPerGrowth', 'AbsFolRate', 'RelFolRate'], axis=1)
	archive.MementoDatetime = pd.to_datetime(archive.MementoDatetime, format='%Y%m%d%H%M%S')
	
	delta_follower = []
	delta_time = []
	delta_pct = []

	for i,fol in enumerate(archive.FollowerCount):
		if i == 0:
			delta = int(0.0)
			time_delta = 1.0
			pct_delta = 0.0
		elif archive.MementoDatetime.iloc[i] == archive.MementoDatetime.iloc[i-1]:
			delta = int(0.0)
			time_delta = 1.0
			pct_delta = 0.0
		else:
			delta = int(fol - archive.FollowerCount.iloc[i-1])
			time_delta = (archive.MementoDatetime.iloc[i] - archive.MementoDatetime.iloc[i-1]).total_seconds() / (60.0*60.0*24.0)
			pct_delta = delta / archive.FollowerCount.iloc[i-1] * 100.0

		delta_follower.append(int(delta))
		delta_time.append(time_delta)
		delta_pct.append(pct_delta)
    
	archive['DeltaFollower'] = delta_follower
	archive['DeltaFollowerPct'] = delta_pct
	archive['PerTime'] = delta_time
	archive['DeltaFollowerPerTime'] = archive.DeltaFollower/archive.PerTime

	archive.columns = ['archive_created_at', 'archive_url', 'follower_count', 'delta_follower', 'delta_follower_pct', 'per_time', 'delta_follower_per_time']
	
	#archive.to_csv('{}/follower_count.csv'.format(thandle)
	archive.to_hdf('{}/follower_count.h5'.format(thandle), key='archive', mode='w')
	print('Done!\n')
	
	return
