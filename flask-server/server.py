
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

print('collecting the data')

# Step 1: Collect Data
df = pd.read_csv('twitter_data.csv')
df = df.drop(['id_str', 'location', 'url', 'description', 'status', 'default_profile', 'default_profile_image', 'has_extended_profile'], axis=1)

# performing feature engineering on followers_count and friends_count columns
df['followers_count'] = df.followers_count.apply(lambda x: 0 if pd.isnull(x) else int(x))
df['friends_count'] = df.friends_count.apply(lambda x: 0 if pd.isnull(x) else int(x))
df['following_to_followers_ratio'] = df['friends_count'] / df['followers_count']

train_df = df.copy()

# performing feature engineering on id and verified columns
# converting id to int
train_df['id'] = train_df.id.apply(lambda x: int(x))
# converting verified into vectors
train_df['verified'] = train_df.verified.apply(lambda x: 1 if ((x == True) or x == 'TRUE') else 0)

# followers are more than following --> then it is most probably a bot
condition = ((train_df.following_to_followers_ratio > 10))  # these all are bots

# converted condition datatype from boolean to int form for easy calculation
train_df['bot'] = condition.astype(int)



# Step 3: Transform Data from text to number format
vectorizer = TfidfVectorizer(stop_words='english')

#change
train_df['screen_name'] = train_df['screen_name'].fillna('')

X = vectorizer.fit_transform(df['screen_name']).toarray()
y = df['bot']

print('training the data')

# Step 4: Train Model
# rf = KNeighborsClassifier(n_neighbors=3)
# rf = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

#calculating accuracy
# accuracy = accuracy_score(X, y)
# print('Accuracy:', accuracy)

# API route
@app.route("/")
@app.route("/checkbot", methods=['GET'])
def checkbot():
    return render_template("index.html")


@app.route("/result", methods = ['POST', 'GET'])
def result():
    output = request.form.to_dict()
    account_name = output["account_name"]

    # Preprocess and transform account description into numerical format
    preprocessed_description = vectorizer.transform([account_name]).toarray()

    # Predict label using trained model
    result = rf.predict(preprocessed_description)

    if result == 1:
        message = " is a bot"
    else:
        message = " is not a bot"

    return render_template("index.html", message=message, account_name=account_name)


if __name__ == "__main__":
    app.run(debug=True)
