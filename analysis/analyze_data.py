import datetime
import random
import warnings
import os

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from pandas.errors import PerformanceWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=PerformanceWarning)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import subprocess

from analysis import process_fch, process_tweets, format_ticks

mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

seed = 42

def get_data(thandle, tweet_file='tweet.js'):
    """
    Scrapes historical follower counts from e.g., the Wayback Machine, and preprocesses the follower count file and Twitter archive for later analysis.
    """
    print('Twitter handle: @{}\n'.format(thandle))
    
    if not os.path.exists(thandle):
        print('Creating new {} directory.\n'.format(thandle))
        os.makedirs(thandle)

    if not os.path.exists('{}/follower_count.csv'.format(thandle)):
        print('Getting historical follower counts. This may take a while!\n')
        subprocess.run(['../fch/__main__.py', '-f', '{}/follower_count.csv'.format(thandle), thandle])
        print('Done!\n')
        
    process_fch.process(thandle)
    
    process_tweets.process(thandle, tweet_file)
    
    return
    
    
def combine_archives(thandle):
    """
    Matches Tweets with coincident follower counts, and creates new variables of interest.
    """
    print('Combining historical follower count and Tweet archives...')
    tweets = pd.read_hdf('{}/tweet_archive.h5'.format(thandle)).reset_index()
    
    followers_archive = pd.read_hdf('{}/follower_count.h5'.format(thandle))

    tweets_followers_archive = pd.DataFrame()
    
    # matching time of Tweet to current follower count
    for i,tweet in tweets.iterrows():
        idx = np.argmin(np.abs(followers_archive.archive_created_at - tweet['created_at']))
        dat = followers_archive.loc[[idx]]
        tweets_followers_archive = tweets_followers_archive.append(dat, ignore_index=True)

    tweets_concat = pd.concat([tweets, tweets_followers_archive], axis=1)
    tweets_concat['favorite_count_log'] = np.log10(tweets_concat['favorite_count'].mask(tweets_concat['favorite_count'] == 0)).fillna(1)
    tweets_concat['retweet_count_log'] = np.log10(tweets_concat['retweet_count'].mask(tweets_concat['retweet_count'] == 0)).fillna(1)
    tweets_concat['favorite_count_per_follower'] = tweets_concat['favorite_count']/tweets_concat['follower_count']
    tweets_concat['retweet_count_per_follower'] = tweets_concat['retweet_count']/tweets_concat['follower_count']
    tweets_concat = tweets_concat.sort_values(by='created_at', ignore_index=True)

    print('Done!\n')
    return tweets_concat


def nclusters_plot(thandle, n_clusters, inertia, k):
	plt.plot(range(1,n_clusters+1), inertia, c='dodgerblue')
	plt.xlabel('k')
	plt.ylabel('Inertia')
	plt.axvline(k, c='k', ls='--')

	plt.tight_layout()
	plt.savefig('{}/inertia_n_clusters.png'.format(thandle), dpi=300)
	plt.close()
	
	
def fav_plot(thandle, K, grouped_clusters, tweets_clip):

	cgen = mpl.cm.viridis([c/(K-1) for c in range(K)])
		
	fig, ax = plt.subplots(1,2, figsize=(12,6))
	
	for c,df in grouped_clusters:
		ax[0].scatter(df.created_at, df.favorite_count, color=cgen[c], s=15, label='Cluster {}'.format(c+1))

	ax[0].set_xlabel('Date')
	ax[0].set_ylabel('$N_\mathrm{Favorites}$')
	ax[0].set_title('Favorites per Tweet')
	ax[0].axhline(0.0, c='k', ls='--')
	ticks = [datetime.datetime.strptime('{}-01-01'.format(y), '%Y-%m-%d') for y in range(tweets_clip.created_at.min().year, tweets_clip.created_at.max().year + 1)]
	tick_labels = [xs.year for xs in ticks]
	if ticks[0] + datetime.timedelta(days=180) < df.created_at.min():
		ticks = ticks[1:]
		tick_labels = tick_labels[1:]
	if ticks[-1] - datetime.timedelta(days=180) > df.created_at.max():
		ticks.append(ticks[-1]+datetime.timedelta(days=365))
		tick_labels.append(tick_labels[-1]+1)
    
	ax[0].set_xticks(ticks)
	ax[0].set_xticklabels(tick_labels)

	ax[0].yaxis.set_major_formatter(tick.FuncFormatter(format_ticks.reformat_large_tick_values))

	for c,df in grouped_clusters:
		ax[1].scatter(df.retweet_count, df.favorite_count, color=cgen[c], s=15, label='Cluster {}'.format(c+1))
            
	ax[1].set_xlabel('$N_\mathrm{Retweets}$')
	ax[1].set_ylabel('$N_\mathrm{Favorites}$')
	ax[1].set_title('Favorites vs. Retweets')
	ax[1].set_xscale('log')
	ax[1].set_xlim(0.5, max(tweets_clip.retweet_count)*1.5)
	ax[1].set_yscale('log')
	ax[1].set_ylim(0.05, max(tweets_clip.favorite_count)*1.5)

	ax[1].legend(loc=4, fontsize=12)

	fig.suptitle('@{}'.format(thandle), fontsize=24)

	plt.tight_layout()
	fig.savefig('{}/clusters.png'.format(thandle), dpi=300)
	plt.close()
	

def cluster_tweets(thandle, tweets_concat, cluster=True):
    """
    Runs clustering algorithm to extract Tweets that are most ad-like.
    """
    from kneed import KneeLocator
    # interested in time period following intial growth, where change in followers between Tweets drops below 0.5%
    #TODO: better way to remove period of initial growth, since 0.5% might not be a good limit for all users
    idx = tweets_concat.where(tweets_concat.delta_follower_pct>0.5).last_valid_index()
    tweets_clip = tweets_concat[idx:]
    	
    if cluster:
        print('Running clustering algorithm. Some human interaction necessary here...')
        # prepare data for clustering algorithm
        # drop variables that aren't necessary for clustering
        tweets_clustering = tweets_clip.drop(['index','id','created_at', # not relevant to extracting ad cluster
                                              'in_reply_to_status_id','in_reply_to_user_id','in_reply_to_screen_name', # use bool instead
                                              'full_text', # use later for sentiment analysis
                                              'hashtags_bool', # use actual hashtags
                                              'user_mentions', # use bool instead
                                              'urls_bool', # use actual url
                                              'media_url','media_type', # link to media (actual photo/video) and media type not relevant to extracting ad cluster
                                              'archive_created_at','archive_url', # not relevant to extracting ad cluster
                                              'follower_count', # follower count at the time of a tweet not relevant to extracting ad cluster
                                              'retweet_count_log', 'favorite_count_log',
                                              'delta_follower', 'delta_follower_pct', # not relevant to extracting ad cluster
                                              'per_time','delta_follower_per_time'# not relevant to extracting ad cluster
                                              ], axis=1)

        # formatting categorical data
        tweets_clustering = tweets_clustering.where(pd.notnull(tweets_clustering), 'FALSE')

        mask = tweets_clustering.applymap(type) != bool
        d = {True: 'TRUE', False: 'FALSE'}

        tweets_clustering = tweets_clustering.where(mask, tweets_clustering.replace(d))

        # identify continuous (i.e., numerical) and categorical data
        num_feats = ['retweet_count', 'favorite_count', 'favorite_count_per_follower', 'retweet_count_per_follower']
        cat_feats = ['is_quote_status', 'hashtags', 'user_mentions_bool', 'urls', 'media_bool', 'in_reply_bool']
        # scale continuous data
        scale = StandardScaler()
        tweets_clustering[num_feats] = scale.fit_transform(tweets_clustering[num_feats])
    
        # One hot encoding/dummy variable encoding of categorical data
        inertia = []

        n_clusters = 30

        for k in range(1,n_clusters+1):
            #TODO: scale data after OHE
            model = KMeans(n_clusters=k, random_state=seed).fit(pd.get_dummies(tweets_clustering))
            inertia.append(model.inertia_)
    
        kn = KneeLocator(range(1,n_clusters+1), inertia, curve='convex', direction='decreasing')
        K = kn.knee
        #print('Optimal number of clusters:', K)

        #TODO: remove from app, just for preliminary check
        
        nclusters_plot(thandle, n_clusters, inertia, K)
    
        #TODO: scale data after OHE
        model = KMeans(n_clusters=K, random_state=seed).fit(pd.get_dummies(tweets_clustering))
        pred = model.labels_
        
        grouped_clusters = tweets_clip.groupby(pred)
    
    
		#TODO: remove from app, just for preliminary check
        # plot the clusters
        fav_plot(thandle, K, grouped_clusters, tweets_clip)
        
		
        # info about the clusters
        for c, df in grouped_clusters:
            print('\n')
            print('====================')
            print('Cluster', c+1, '(Number of Tweets: {})'.format(len(df)))
            print('Most tweets are replies?', df['in_reply_bool'].mode().tolist()[0])
            print('Most tweets include media?', df['media_bool'].mode().tolist()[0])
            print('Hashtags:')
            for i, count in enumerate(df['hashtags'].value_counts()[:4]):
                if count/len(df) > 0.015:
                    print('    ', df['hashtags'].value_counts()[:4].index.tolist()[i])
            print('URLS:')
            for i, count in enumerate(df['urls'].value_counts()[:3]):
                if count/len(df) > 0.005:
                    print('    ', df['urls'].value_counts()[:3].index.tolist()[i])
            print('N_Favorites', int(df['favorite_count'].median()))
            print('N_Retweets', int(df['retweet_count'].median()))
            print('====================')
        
        ad_cluster = int(input('\nBased on the table above, which cluster seems to be mostly made up of ad Tweets?\nCluster '))
        print('Thanks!\n')
        
        ads = grouped_clusters.get_group(ad_cluster-1)
        
        return ads
    
    else:
        return tweets_clip
    
    
def sig_follower_change(tweets_concat, ads):
    """
    Finds ads that are outliers (i.e., causing a significant increase/decrease in followers).
    """
    print('Finding ads that caused significant follower losses/gains...')
    # calculate median change in followers over the 7 days after a Tweet
    ad_follower_change = []
    for i,ad in enumerate(ads['created_at']):
        days = 7
        time_delta = ad + datetime.timedelta(days=days)

        loid = tweets_concat.index[tweets_concat['created_at'] == ad].to_list()[0]
        hiid = tweets_concat['created_at'].loc[(tweets_concat['created_at'] > ad) & ((tweets_concat['created_at'] <= time_delta))]
        if len(hiid) == 0:
            hiid = tweets_concat['created_at'].loc[(tweets_concat['created_at'] > ads['created_at'].iloc[i-1]) & ((tweets_concat['created_at'] <= time_delta))]
            
        hiid = hiid.iloc[[-1]].index.to_list()[0]
    
        ad_follower_change.append(tweets_concat['delta_follower_per_time'].iloc[loid:hiid].median())
    
    ads.loc[:, 'ad_follower_change'] = ad_follower_change
    ads.dropna(subset = ['ad_follower_change'], inplace=True)

    ols = LinearRegression() # create LinearRegression object
    x_fit = np.asarray(pd.to_datetime(ads.created_at).astype(int).astype(float)).reshape(-1, 1) # use created_at as feature
    y_fit = np.asarray(ads.ad_follower_change) # use median change in followers as target
    ols.fit(x_fit, y_fit) # fit the OLS model
    slope_ols = ols.coef_[0]
    intercept_ols = ols.intercept_

    y_pred = intercept_ols + x_fit*slope_ols

    lossy_tweets = []
    gainy_tweets = []
    for i,ad in enumerate(ads['ad_follower_change']):
        #TODO: check tweets over the last day that might also have cause significant changes
        if (ad - y_pred[i])[0] > 2.0*np.std(y_fit):
            gainy_tweets.extend(ads['full_text'].iloc[i-2:i].to_list())
        if (ad - y_pred[i])[0] <= -1.0*np.std(y_fit):
            lossy_tweets.extend(ads['full_text'].iloc[i-2:i].to_list())
      
    print('Done!\n')
    return gainy_tweets, lossy_tweets


def report(thandle, gain_tweets, loss_tweets):
    """
    Write report.
    """
    from fpdf import FPDF
    import textwrap
	
    print('Writing report...')
    pdf = FPDF()

    # Add a page
    pdf.add_page()
  
    # set style and size of font
    # that you want in the pdf
    pdf.set_font("Arial", size = 16)
  
    # create a cell
    pdf.cell(200, 10, txt = 'Ads that likely caused some loss of Followers:\n\n',
         ln = 1, align = 'L')
    pdf.cell(200, 10, txt = '',
         ln = 2, align = 'L')

    # add another cell
    pdf.set_font("Arial", size = 14)
    tweet_string = ''
    fact = 3
    count = 0
    tweets = []
    while count < fact:
        random.shuffle(loss_tweets)
        rand_tweet = loss_tweets[-1]
        tweets.append(rand_tweet)
        mask = [(tweet != rand_tweet) for tweet in loss_tweets]
        loss_tweets = [loss_tweets[i] for i in range(len(loss_tweets)) if mask[i]]
        count += 1
    
    for i,d in enumerate(tweets):
        string_encode = d.encode('ascii', 'ignore')
        string_decode = string_encode.decode()
        d = textwrap.wrap(string_decode, width=65)

        for j in d:
            pdf.cell(200, 10, txt=j, ln = fact, align = 'L')
            fact += 0.5
        pdf.cell(200, 10, txt='', ln = fact, align = 'L')
        fact += 1
        

    pdf.output('{}/lossy_tweets.pdf'.format(thandle))
    pdf.close()
    
    pdf = FPDF()

    dupes_gain = set([x for n, x in enumerate(gain_tweets) if x in gain_tweets[:n]])


    # Add a page
    pdf.add_page()
  
    # set style and size of font
    # that you want in the pdf
    pdf.set_font("Arial", size = 16)
  
    # create a cell
    pdf.cell(200, 10, txt = 'Ads that likely caused some gain in Followers:\n\n',
         ln = 1, align = 'L')
    pdf.cell(200, 10, txt = '',
         ln = 2, align = 'L')
  
    # add another cell
    pdf.set_font("Arial", size = 14)
    tweet_string = ''
    fact = 3
    count = 0
    tweets = []
    while count < fact:
        random.shuffle(gain_tweets)
        rand_tweet = gain_tweets[-1]
        tweets.append(rand_tweet)
        mask = [(tweet != rand_tweet) for tweet in gain_tweets]
        gain_tweets = [gain_tweets[i] for i in range(len(gain_tweets)) if mask[i]]
        count += 1
    
    for i,d in enumerate(tweets):
        string_encode = d.encode('ascii', 'ignore')
        string_decode = string_encode.decode()
        d = textwrap.wrap(string_decode, width=65)

        for j in d:
            pdf.cell(200, 10, txt=j, ln = fact, align = 'L')
            fact += 0.5
        pdf.cell(200, 10, txt='', ln = fact, align = 'L')
        fact += 1
        
    pdf.output('{}/gainy_tweets.pdf'.format(thandle))
    pdf.close()
    
    print('Done!\n')
    
