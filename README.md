# Beyond Twitter Analytics with Unsupervised Machine Learning



Twitter Analytics can provide useful insights for users by providing, for example, detailed tracking of impressions and engagements, number of Followers, and changes in these markers following the use of specific hashtags and/or a promoted Tweet. However, Twitter Analytics does not provide users with information about what specific Tweet content (text and sentiment, the inclusion of hashtags, urls, or media) impacts a Follower’s decision to favorite, retweet, or even unfollow the user. In particular, for small businesses, understanding the immediate impact of an advertisement on their Followers’ opinions of that business can help improve the composition of future ads to maximize engagements and prevent a decrease in Followers.

## Data & Analysis

As an example, I have scraped Follower counts from web archives—including the Wayback Machine and the Stanford Web Archive Portal—and all Tweets from the popular account WeRateDogs (`@dog_rates`). Ideally, however, the owner of an account would just download an archive of their Twitter data for use. Given these data for WeRateDogs ([Follower counts here](dog_rates_follower_count.h5) and [Tweets here](dog_rates_tweets_archive.h5), approx. 8 MB total), I run an unsupervised clustering algorithm on the continuous (number of favorites and retweets counts, and number of favorites per Follower) and categorical data (hashtags, user mentions, urls, media type, and whether the Tweet is a reply to a user or a retweet) to extract advertisements.

For the WeRateDogs Tweet archive, Tweets are assigned to 6 clusters. The clusters are differentiated primarily by the number of favorites and reteweets each Tweet receives, but also depend on whether a Tweet included a link to a product page or a fundraiser, or a specific hashtag.    

![WeRateDogs Tweet clustering](https://github.com/hmlewis-astro/twitter_insights/blob/master/dog_rates_clusters.png)

Following each advertisement, I then track changes in the number of Followers over the next 7 days, to see if any one ad has caused a significant number of Followers to unfollow the account.

## Results

For ads Tweeted by WeRateDogs, this initial analysis shows that some Tweets revolving around social and political topics, namely Tweets about racial issues or COVID-related material, result in a decrease in followers (see a random sampling of these Tweets [here](https://github.com/hmlewis-astro/twitter_insights/blob/master/dog_rates_lossy_tweets.pdf)). My planned work for this project include (1) performing sentiment analysis on Tweet text in order to track changes in the number of Followers following any tweets perceived as strongly positive or negative and (2) creating a web app that will allow any user to upload their own Twitter archive to obtain insights into the types of advertisements and general Tweets that gain and lose followers. This web app will be applicable to e.g., small businesses, social media influencers, and politicians, all of whom are actively, constantly trying to grow their brands and Follower base.

## Usage example

Clone the repository.
To run via the Jupyter Notebook: 
```sh
jupyter notebook haberman_analysis.ipynb
```
Run all cells (select "Cell" > "Run All" or press Shift+Return to run individual cells).

To run via the Command Line:
```sh
python twitter_data_impact.py 
```
When prompted, enter the cluster number that appears to have the most advertisement-like Tweets (includes link to product website/fundraisers, may use the hashtag `ad`, etc.)


These scripts will produce the plot shown above, as well as a plot of the inertia of the clustering algorithm for an increasing number of clusters; this [plot](https://github.com/hmlewis-astro/twitter_insights/blob/master/dog_rates_inertia_n_clusters.png) is used to determine the optimal number of clusters for the clustering algorithm. A [PDF](https://github.com/hmlewis-astro/twitter_insights/blob/master/dog_rates_lossy_tweets.pdf) containing a sampling of the advertisements that caused a significant number of Followers to unfollow the account is also created.

## Author

Hannah Lewis – hlewis@virginia.edu

Distributed under the MIT License. See ``LICENSE`` for more information.

[https://github.com/hmlewis-astro](https://github.com/hmlewis-astro)
