import numpy as np
import pandas as pd
from urllib.parse import urlparse

def process(thandle, tweet_file):
	print('Formatting Tweet archive...')
	tweets = pd.read_json(tweet_file, orient='index')
	
	tweets.index = pd.to_numeric(tweets.index, downcast='integer')
	tweets.created_at = tweets.created_at.dt.tz_convert(None)
	
	tweets['hashtags'] = [np.nan] * len(tweets)
	tweets['hashtags_bool'] = [False] * len(tweets)
	hashtags = []
	hashtags_bool = []
	for i,row in enumerate(tweets['entities']):
		if row['hashtags'] == []:
			hashtags.append(np.nan)
			hashtags_bool.append(False)
		else:
			tags = []
			for j,tag in enumerate(row['hashtags']):
				tags.append(tag['text'])

			if len(tags) == 1:
				tags = tags[0]
			else:
				tags = [t if (('ad' or 'partner' or 'Partner') in t) else tags[0] for t in tags][0]

			hashtags.append(tags)
			hashtags_bool.append(True)

	tweets['hashtags'] = hashtags
	tweets['hashtags_bool'] = hashtags_bool
	
	tweets['user_mentions'] = [np.nan] * len(tweets)
	tweets['user_mentions_bool'] = [False] * len(tweets)
	mentions = []
	mentions_bool = []
	for i,row in enumerate(tweets['entities']):
		if row['user_mentions'] == []:
			mentions.append(np.nan)
			mentions_bool.append(False)
		else:
			ment = []
			for j,men in enumerate(row['user_mentions']):
				ment.append(men['screen_name'])
			mentions.append(ment)
			mentions_bool.append(True)

	tweets['user_mentions'] = mentions
	tweets['user_mentions_bool'] = mentions_bool
	
	tweets['urls'] = [np.nan] * len(tweets)
	tweets['urls_bool'] = [False] * len(tweets)
	urls = []
	urls_bool = []
	for i,row in enumerate(tweets['entities']):
		if row['urls'] == []:
			urls.append(np.nan)
			urls_bool.append(False)
		else:
			url_list = []
			for j,u in enumerate(row['urls']):
				url_parse = urlparse(u['expanded_url']).netloc.replace('www.','').replace('.com','').replace('.co','').replace('.org','')
				if ('gofundme' or 'gf.me') in url_parse:
					url_parse = 'gofundme'
				url_list.append(url_parse)
        
			urls.append(url_list[0])
			urls_bool.append(True)
            
	tweets['urls'] = urls
	tweets['urls_bool'] = urls_bool
	
	tweets['media_url'] = [np.nan] * len(tweets)
	tweets['media_type'] = [np.nan] * len(tweets)
	tweets['media_bool'] = [False] * len(tweets)
	media = []
	media_type = []
	media_bool = []
	for i,row in enumerate(tweets['entities']):
		#if row['media'] == []:
		if len(row) < 5:
			media.append(np.nan)
			media_type.append(np.nan)
			media_bool.append(False)
		else:
			for j,m in enumerate(row['media']):
				med = m['expanded_url']
				med_type = m['type']
			media.append(med)
			media_type.append(med_type)
			media_bool.append(True)

	tweets['media_url'] = media
	tweets['media_type'] = media_type
	tweets['media_bool'] = media_bool
	
	tweets['in_reply_bool'] = np.where(tweets.in_reply_to_status_id == tweets.in_reply_to_status_id, True, False)

	tweets = tweets.drop('entities', axis=1)
	
	#tweets.to_json('{}/tweet_archive.js'.format(thandle))
	tweets.to_hdf('{}/tweet_archive.h5'.format(thandle), key='tweets', mode='w')

	print('Done!\n')
	
	return
