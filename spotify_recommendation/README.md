### Spotify Recommendation System

Goal of this project is to create a content-based filtering Spotify Song Recommendation System.

Have you ever thought how “Recommended (based on what’s in your playlist)” on your Spotify works? This project helps answer this question and walks through the process of building machine learning pipeline that can help predict what user would like to listen to next.

#### Executive summary

Spotify is a platform that makes money through end users via subscriptions. Spotify song recommendation system will help user discover engaging content to increase DAU (daily active user) metrics. We will use machine learning to filter out the content and present engaging content to the user for increasing their product usage.

#### Rationale

If users are unable to discover good content, they will move to other competitive platforms for finding music. Good recommendations will help with user retention and in-turn higher revenues for the company.

#### Data Sources

I first started by taking a look at the dataset in [Kaggle](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset). The dataset was organized in 3 different files

 - data.csv
 - data_by_genres.csv
 - data_by_year.csv
 
Next, I wanted to explore the spotify's Web API to see if I can extract my personal playlists data. The base URI for all Web API requests is https://api.spotify.com/v1 and spotipy wraps this up neatly for us to use. 

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

To do this, I first examinded the data to see if there are is any NULL data

![](images/year_non_null.png)
![](images/genre_non_null.png)
![](images/data_non_null.png)

By quick examination following short list of categories could be setup:

- Metadata
	- id
   	- genres
   	- artist_pop
   	- track_pop
- Audio
	- Mood: Danceability, Valence, Energy, Tempo
	- Properties: Loudness, Speechiness, Instrumentalness
	- Context: Liveness, Acousticness
	- metadata: key, mode

- Text
	- track_name
	

After data preprocessing, we can work on "Feature Generation" using a pipeline for feature generation. Methods like 'One-hot Encoding' and 'Normalization' would be useful for this step.

The next step is to perform content-based filtering based on the song features we have. To do so, we concatenate all songs in a playlist into one summarization vector. Then, we find the similarity between the summarized playlist vector with all songs (not including the songs in the playlist) in the database. Then, we use the similarity measure retrieved the most relevant song that is not in the playlist to recommend it.

There are three steps in this section:

- Choose playlist: Retrieve a playlist
- Extract features:Retireve playlist-of-interest features and non-playlist-of-interest features
- Find similarity: Compare the summarized playlist features with all other songs.


#### Results

Using the content-based filtering system, we are able to help end users discover new engaging content. 

#### Next steps

Success for this could be measured by segmenting users into 2 groups. Group A could be users who are provided these recommendations, and Group B could be users who are not provided these recommendations. Next, we could compare daily active usage metrics (eg. minutes spent by users on Spotify) across these two groups for a few months to understand if this this recommendation system is useful or not for achieving the business goals.

##### Contact and Further Information

Akash Choudhari
akashtc@gmail.com