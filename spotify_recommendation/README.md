### Spotify Recommendation System

Goal of this project is to create a content-based filtering Spotify Song Recommendation System. As part of creating this system, one of the essential part was to understand how Spotify understands ‘popularity’. Have you ever thought how “Recommended (based on what’s in your playlist)” on your Spotify works? This project helps answer this question and walks through the process of building machine learning pipeline that can help predict what user would like to listen to next.

#### Executive summary

Spotify is a platform that makes money through end users via subscriptions. Spotify song recommendation system will help user discover engaging content to increase DAU (daily active user) metrics. We will use machine learning to filter out the content and present engaging content to the user for increasing their product usage.

#### Rationale

If users are unable to discover good content, they will move to other competitive platforms for finding music. Good recommendations will help with user retention and in-turn higher revenues for the company.

#### Data Sources

I first started by taking a look at the dataset in [Kaggle](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset). The dataset was organized in 3 different files

 - data.csv
 - data_by_genres.csv
 - data_by_year.csv

Just for additinal context, some of the fields in this dataset are explained below:

 - Instrumentalness: This value represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is

 - Acousticness: This value describes how acoustic a song is. A score of 1.0 means the song is most likely to be an acoustic one

 - Liveness: This value describes the probability that the song was recorded with a live audience. According to the official documentation “a value above 0.8 provides strong likelihood that the track is live”

 - Speechiness: “Speechiness detects the presence of spoken words in a track”. If the speechiness of a song is above 0.66, it is probably made of spoken words, a score between 0.33 and 0.66 is a song that may contain both music and words, and a score below 0.33 means the song does not have any speech

 - Energy: “(energy) represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy”

 - Danceability: “Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable”

 - Valence: “A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)”.

Next, I wanted to explore the spotify's Web API to see if I can extract my personal playlists data. The base URI for all Web API requests is https://api.spotify.com/v1 and spotipy wraps
this up neatly for us to use.

Spotify publishes some of the fields below which will be useful to understand similarities between songs:

 - acousticness number <float>
 A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic with value >= 0 and <= 1

 -  analysis_url string
 A URL to access the full audio analysis of this track. An access token is required to access this data.

 - danceability number <float>
 Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.

 - duration_ms integer
 The duration of the track in milliseconds.

 - energy number <float>
 Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

 - id string
 The Spotify ID for the track.

 - instrumentalness number <float>
 Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

 - key integer
 The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. The values will be >= -1 and <= 11
 - liveness number <float>
 Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

 - loudness number <float>
 The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.

 - mode integer
 Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

 - speechiness number <float>
 Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

 - tempo number <float>
 The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

 - time_signature integer
 An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4" with value >= 3 and <= 7
 - track_href string
 A link to the Web API endpoint providing full details of the track.

 - type string
 The object type.

 - audio_features uri string
 The Spotify URI for the track.

 - valence number <float>
 A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry) with value >= 0 and <= 1

#### Methodology

First we have to do some data preprocessing for the imported data. As part of data preprocessing, we will find what data is useful and convert some data (eg. genres) to lists for easier operations. As part of useful data selection, we will first drop duplicates in our dataset. Then, we could look into which features are important.

To do this, I first examined the data to see if there are is any NULL data. Follow along my juptyer notebook for these steps [here](Spotify_Recommendation_System.ipynb)

![](images/year_non_null.png)
![](images/genre_non_null.png)
![](images/data_non_null.png)

I did clean up for data by stripping unnessary characters and removing non-english characters

![](images/clean_chars.png)
![](images/clean_non_english.png)

As seen, there are some zero values in the tempo column, but none in the year or duration.

![](images/non_zero_values.png)

I fixed the tempo values by imputing the values.

![](images/impute_tempo_values.png)

Next I looked at the correlation between different datasets. From this graph we can start to make connections. E.g, popularity and year have a strong connection. Songs that come out that year have high popularity. Energy and loudness have a strong connection, as we can assume that more loudness = more energy. Danceability and valence are also fairly high, as valence is the "happy" level in music. Dancing is happy!

![](images/data_corr_.png)

I also looked at the feature correlation

![](images/feature_corr.png)

I looked at the Music data over time. I grouped the data by year to understand the distribution better. As you can see most of the songs are from 1950-2010 and are fairly evenly distrubuted

![](images/music_distribution.png)

Further, I reviewed some of the sound features. I ploted them to see how those trends evolved or changed over time. As you can see acousticness and instrumentalness decreased over time, but energy and danceability has trended upwards. These trends may be helpful in predictions.

![](images/sound_distribution.png)

This dataset contains the audio features for different songs along with the audio features for different genres. We can use this information to compare different genres and understand their unique differences in sound.

![](images/genre_distribution.png)

Next, I did some analysis to understand the data. A

As you can see, upwards trend in danceability from 1920 until a downward trend from approximately 1930-35. would assume because of the war. This graph shows an upward trend in danceability from about 1945-1950 onwards... coinciding with the end of WWII and the onset of the 60. Fast upward trend from the 1950s onward, probably from the end of WWII and the ability to make more music, as well as people being able to listen to music, AND the steep jump in the early 2000s as a result of streaming.

![](images/year_popularity_.png)

Interesting...some songs are loud but low energy, and fewer still that are high energy but quieter than average as seen at the top and tail ends of the graph. The general consensus is that as the songs increase in loudness, the energy increases.

![](images/energy_loudness.png)

I also took a closer look at top 10 tracks and artists.

![](images/top_10_artists.png)
![](images/top_10_songs.png)

Another data point I wanted to check was how may songs are released per year, and whether that data is skewed for specific years.

![](images/songs_per_year.png)

For the top generes, I checked if there is specific pattern around audio features

![](images/top_10_genres.png)

Next I wanted to check if there is specific clustering of data. For this I used the KMeans clustering algorithm. and fitted a pipeline and plotted it for genres.

![](images/kmeans_genres.png)

The essential question for me was: could we use a song’s attributes to predict a track’s ‘popularity’, so I looked into building some models and understanding how this could be predicted well. For this I built two models. In next sectiion I set up the models and do GridSearch to find the best parameters.

Next we examine if there is any specific signatures for songs too.

![](images/kmeans_songs.png)

##### Building models and Gridsearch for parameters

Lets just first fit DecisionTree algorithm. As you can see the mean square error is pretty high with default parameters.

![](images/dtree_default_params.png)

Using the GridSearchCV, we fitted the model again to reduce the mean square error


![](images/dree_gridsearch.png)

The resulting dtree graph is below

![](images/dtree_improved_score.png)
![](images/tree.eps)

The other model I was to use KNeighborsRegressor. The steps below show the model and hyperparameter tuning done for this model

![](images/kNN_Model.png)

##### Million Songs database

The million song dataset is a very popular dataset and is available at https://labrosa.ee.columbia.edu/millionsong/. The original dataset contained quantified audio features of around a million songs ranging over multiple years.

Follow along this analysis in [Million_Songs_data_gathering.ipynb](Million_Songs_data_gathering.ipynb)

The two datasets can download on Kaggle. I have also uploaded it for reference [song_data.csv](data/song_data.csv.zip) and [count play of 10,000 songs](data/10000.txt.zip). We can capture this into pandas dataframe from triplet_file and metadata_file. The triplet_file of 2,000,000 rows contains user_id, song_id and listen count. The metadata_file of 1,000,000 rows of songs contains song_id, title, release_by (release date) and artist_name. Next steps, would be to merge the two datasets by song_id, bringing the count of unique songs from 1,000,000 to 10,000 songs.

The next part of the dataset was called from the Spotify Web API. To do this I used an open source tool (https://github.com/spotDL/spotify-downloader) which can convert a list of songs into a Spotify playlist. I exported a CSV of the 10,000 song names from our previous dataset and created Spotify playlists of around 7k same songs. The other 3k songs were not found in the Spotify search. Using the Spotify API, I extracted the artist name, song title, and spotify ID of the tracks of these playlists. I then merged this data set with our previous dataset by song title and artist and stored it in a new file - [User_SongFeatures_data](data/User_SongFeatures_data.csv) which was compressed using "xz" format since Github doesnt allow files >100MB to be uploaded. The last part of the dataset are audio features also called from the Spotify API. The audio features were extracted using the unique spotify ID’s of our dataset and then were added in as additional columns.

Below table explains the various values and their description.

| KEY |   VALUE DESCRIPTION |
|-----|--------------------|
|duration_ms |The duration of the track in milliseconds|
|key |The estimated overall key of the track|
|mode |Mode indicates the modality of a track. Major is represented by 1 and minor is 0|
|time_signature |The time signature is a notational convention to specify how many beats are in each bar|
|acousticness |A confidence measure from 0.0 to 1.0 of whether the track is acoustic|
|danceability |Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity|
|energy |Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity|
|instrumentalness |Predicts whether a track contains no vocals|
|liveness |Detects the presence of an audience in the recording|
|loudness |The overall loudness of a track in decibels (dB)|
|speechiness |Speechiness detects the presence of spoken words in a track|
|valence |A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track|
|tempo |The overall estimated tempo of a track in beats per minute (BPM)|



###### Exploratory Data Analysis for merged dataset

Next lets explore the dataset consisting of user's listening activity. In this EDA we explore the dataset without the Spotify audio features. The purpose of this analysis is to understand the users listening behavior and see if there are any outliers that may skew our recommendations or make our dataset invalid. There are four particular areas we will explore:

Follow along this analysis in [Exploratory_Data_Analysis_for_merged_dataset.ipynb](Exploratory_Data_Analysis_for_merged_dataset.ipynb)

- How many different songs does a user actually listen to?
- How many times is each song listened to?
- How many unique users listen to each song?
- How many different songs does a user actually listen to?

First we explore the distribution of how many different songs each user listens to. The purpose of this is to make sure that our data makes sense with real life user listening habits and to also make sure the data is valid enough to train our recommendation machine.

Number of unique songs a user listens to is shown below
![](images/unique_songs_per_listener_.png)

It does appear that there are quite a few outliers in regards to the amount of songs a user listens to. In this case, what we are concerned about is having a dataset that would not allow us to train a recommendation system. In our data, only 3372 out of 74899 users only listened to one song (4.5%) so we can conclude that our data is valid for our model.

- How many times is each song listened to?

Next we look at the number of times each song is listened to by users. We want to see if there are any outliers so that our analysis is not skewed when recommending songs. For example, we wouldn't want a song to be recommended to users only because it's listen count is considerably higher than all other songs.

From the boxplot you can see that there are a few songs with very high listen counts. Below is a closer look at the songs with a listen count greater than 10,000.

![](images/box_plot_number_of_times.png)

These songs would be considered outliers, however we do not want to remove them from our data. But we do want to keep these songs in mind during our modeling stage so that they will not be recommended to users solely due to their popularity.

Figure below shows the distribution of number of times a song is listened to.

![](images/dist_number_of_times.png)

I also examined most popular songs (>10k listens)

![](images/Songs_gt_10k.png)

- How many unique users listen to each song?

Next we looked at the distribution of the number of unique users that listen to each song. We explored this to see if the songs with high listen counts were caused by certain users listening to these songs an extreme amount of times as this could potentially skew our data. The scatterplot shows that there is a relationship between number of listens and number of users. With the exception of a couple of outliers, the number of users that listen to a song increases as the number of listens increases. This tells us that the listen count of our popular songs is not skewed by a few users with extreme listen counts.

![](images/Scatterplot_unique_users.png)

- Whats the total listen count of each user?

Now we want to see the distribution of the amount of times a user listens to any song (total listen count). Again, if a large amount of the users also listen once this could make our dataset not usable for training a recommendation engine.

![](images/listen_count_per_user.png)

See below for box plot of listen count

![](images/box_plot_total_listen_count.png)

Only 2065 users have a listen count of one. This is a low percentage, making our dataset valid for building our model.

For further analysis, I explored the data to see how listen count evolved over the decade

![](images/listen_count_per_decade.png)

I also compared some important features to listen count

![](images/danceability_vs_listen_count.png)

![](images/valence_vs_listen_count.png)

I also looked at histogram of important features to understand if these are any big anomalies.

![](images/hist_danceability.png)

![](images/hist_top_energy.png)

![](images/hist_top_songs.png)


######  Key Findings

This analysis tells us that we have a valid dataset to begin training our recommendation engine. There are some songs that are 'outliers' in regards to listen count however we will not remove these outliers. We will just keep these songs in mind during the modeling stage and make sure they are not recommended to a lot of users solely due to their popularity.

#### Recommendation System

######  Approach

We will use a combination of matrix factorization and classification to produce song recommendations for a particular user. We will perform matrix factorization on a subset of our data to extract latent features for our users and songs. Then, we will use these latent factors in conjunction with our audio features to train a classification model. The model will predict classes of a high listen count versus a low listen count of a song. These predictions will then be used to recommend songs to a particular user.

For our model we will randomly split the dataset into three sets. There will be two test data sets and one validation dataset. The first data set will be used to perform matrix factorization to extract user and item latent factors. The second dataset will be used to train our classification model. And lastly, our validation set will be used to evaluate our model.

######  Matrix Factorization

Matrix factorization is to find out two matrices such that when you multiply them you will get back the original matrix. Matrix factorization can be used to discover latent features underlying the interactions between two different kinds of entities. The purpose of performing matrix factorization in our project is to extract latent factors of the users and the songs. Be
low is a matrix of our user-item pairs.

The intuition behind using matrix factorization is that there should be some latent features that determine how many times a user listens to a song. For example, two users would listen to the same song if they both like the genre of the song. Therefore if we can discover these latent features, we can add them as additional features to our dataset since the features associated with the user should match with the features associated with the song.

The images above show the values of the latent factors for the user (u) and song (s). These factors will now be used in our classification model. Although we already have some song features from Spotify, the latent factors of the user should help the strength of our recommendation system.

######  Classification

After adding on the latent factors found from matrix factorization, we then perform classification on the data set, ignoring the user_id and song_id columns. The column ‘listen_count’ is transformed into classes of ‘one’ and ‘one_plus’. Since the listen counts are highly skewed we will only perform a binary classification.

###### Evaluation Metrics

To evaluate our model we will use ROC AUC score because we have a binary classification with skewed classes. AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents measure of separability. It tells how much model is capable of distinguishing between classes.
The image below shows an example of the ROC curve. The shaded area is AUC (Area Under Curve) and the axis’ represent False Positive and True Positive Rates.

In our case, since we have two classes our baseline AUC is 0.5. If our AUC is 0.5, this tells us our model has no capability of distinguishing between our two classes. We aim to get the AUC of our model as close to 1 as possible.

###### XGBoost

We first train an XGBoost model on our training set. Without tuning the hyperparameters, the base model gives us an AUC score of 0.6026. After evaluating the feature importance of the model and tuning subsample, colsample_bytree, max_depth, learning_rate and n_estimators, our XGBoost model gives us an AUC score of .657. This AUC tells us that XGBoost has a 66% chance of distinguishing between our two classes of high listen counts and low listen counts.

###### Feature Importance

Now that we have extracted these latent factors from matrix factorization. Let’s see how much of an effect these features have on our model. To do this, we use the feature_importances_ feature of our trained XGBoost model.

You can see that the user latent factors have very high ‘importance’ in comparison to our other features. However, our song or item latent factors have low ‘importance’ in compared to our other features. This could be because we already have features of the songs so the additional song latent factors do not add any value.
The user latent factors hold a lot of weight when predicting. This makes sense in relation to our final objective of recommending songs to a user. Of course factors of a user are important when choosing songs to recommend to that user. In this model we dropped the 9 least important features.

###### Random Forest

To see if we can increase the models score, we train a Random Forest classifier to be ensembled with our XGBoost model. After training the Random Forest model and increasing the number of trees, the classifier gives us an AUC score of .669. Random Forest has a 67% chance of correctly predicting between our two classes. This score is higher than XGBoost so we then ensemble the two models by averaging their predictions and see if it produces an even higher AUC score.

####### Ensemble

To create a final model that uses both our trained XGBoost and Random Forest classifiers, we take an average of the predicted probabilities of the two classes. The averaged probability is then used to predict a class.
This ensembled model gives us an AUC score of .68. This is a .07 increase from our base XGBoost model. This AUC tells us that our model has a 68% chance of correctly distinguishing between our two classes.

#### Results

At prediction time, if we want to know if a user will listen to a song we will join the user features and the song features of that song and predict. The function ‘get_top_songs’, takes in a user id as an argument and recommends five songs by returning the five songs with the highest probability of belonging to our class representing a high listen count.

Above shows the recommended songs for user ‘f1ccb26d0d49490016747f6592e6f7b1e53a9e54'. Besides AUC score, another way we can evaluate the recommendation system is by seeing if recommended songs are similar to what that user has listened to. Below are the users top 5 listened to songs. How did we do?

#### Next steps

This type of model requires a ton of data. The number of user and item pairs is very large so we are training on a very small subset of the universe of possibilities. More data is necessary for a better score. It is difficult to create a good recommender system with a small amount of data.

Although not big enough, the dataset is still very large. The size of the data affects the quality of hyperparameter tuning of the model. Time and computing power limits our ability to use a grid search method for tuning our hyperparameters. This method would most likely have returned a better final AUC score.

##### Contact and Further Information

Akash Choudhari
akashtc@gmail.com