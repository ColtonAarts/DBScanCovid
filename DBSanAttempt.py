from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from os import walk
import json
import datetime
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import Util
import numpy as np
from nltk import PorterStemmer
from KmeansClass import Custom
from nltk import TweetTokenizer
from json import JSONDecodeError

dates = ([datetime.datetime(2020, 8, 3, 0, 0), datetime.datetime(2020, 8, 2, 0, 0), datetime.datetime(2020, 8, 1, 0, 0), datetime.datetime(2020, 7, 31, 0, 0), datetime.datetime(2020, 7, 27, 0, 0), datetime.datetime(2020, 7, 30, 0, 0), datetime.datetime(2020, 7, 28, 0, 0), datetime.datetime(2020, 8, 9, 0, 0), datetime.datetime(2020, 8, 8, 0, 0), datetime.datetime(2020, 8, 7, 0, 0), datetime.datetime(2020, 8, 6, 0, 0), datetime.datetime(2020, 8, 5, 0, 0), datetime.datetime(2020, 8, 4, 0, 0), datetime.datetime(2020, 8, 10, 0, 0), datetime.datetime(2020, 8, 11, 0, 0), datetime.datetime(2020, 8, 12, 0, 0), datetime.datetime(2020, 8, 13, 0, 0), datetime.datetime(2020, 8, 17, 0, 0), datetime.datetime(2020, 8, 16, 0, 0), datetime.datetime(2020, 8, 15, 0, 0), datetime.datetime(2020, 8, 14, 0, 0), datetime.datetime(2020, 8, 18, 0, 0), datetime.datetime(2020, 8, 19, 0, 0), datetime.datetime(2020, 8, 20, 0, 0), datetime.datetime(2020, 8, 21, 0, 0), datetime.datetime(2020, 8, 24, 0, 0), datetime.datetime(2020, 8, 23, 0, 0), datetime.datetime(2020, 8, 22, 0, 0), datetime.datetime(2020, 8, 25, 0, 0), datetime.datetime(2020, 8, 26, 0, 0), datetime.datetime(2020, 8, 27, 0, 0), datetime.datetime(2020, 8, 28, 0, 0), datetime.datetime(2020, 8, 29, 0, 0), datetime.datetime(2020, 8, 30, 0, 0), datetime.datetime(2020, 8, 31, 0, 0), datetime.datetime(2020, 9, 1, 0, 0), datetime.datetime(2020, 12, 2, 0, 0), datetime.datetime(2020, 12, 1, 0, 0), datetime.datetime(2020, 11, 30, 0, 0), datetime.datetime(2020, 11, 29, 0, 0), datetime.datetime(2020, 11, 27, 0, 0), datetime.datetime(2020, 12, 3, 0, 0), datetime.datetime(2020, 12, 7, 0, 0), datetime.datetime(2020, 12, 6, 0, 0), datetime.datetime(2020, 12, 5, 0, 0), datetime.datetime(2020, 12, 4, 0, 0), datetime.datetime(2020, 11, 28, 0, 0), datetime.datetime(2020, 12, 8, 0, 0), datetime.datetime(2020, 12, 9, 0, 0), datetime.datetime(2020, 12, 10, 0, 0), datetime.datetime(2020, 12, 11, 0, 0), datetime.datetime(2020, 12, 12, 0, 0), datetime.datetime(2020, 12, 14, 0, 0), datetime.datetime(2020, 12, 13, 0, 0), datetime.datetime(2020, 12, 15, 0, 0), datetime.datetime(2020, 12, 16, 0, 0), datetime.datetime(2020, 12, 17, 0, 0), datetime.datetime(2020, 12, 18, 0, 0), datetime.datetime(2020, 12, 19, 0, 0), datetime.datetime(2020, 7, 2, 0, 0), datetime.datetime(2020, 7, 1, 0, 0), datetime.datetime(2020, 6, 30, 0, 0), datetime.datetime(2020, 6, 28, 0, 0), datetime.datetime(2020, 6, 25, 0, 0), datetime.datetime(2020, 6, 29, 0, 0), datetime.datetime(2020, 7, 3, 0, 0), datetime.datetime(2020, 7, 4, 0, 0), datetime.datetime(2020, 7, 5, 0, 0), datetime.datetime(2020, 7, 6, 0, 0), datetime.datetime(2020, 7, 7, 0, 0), datetime.datetime(2020, 7, 8, 0, 0), datetime.datetime(2020, 7, 9, 0, 0), datetime.datetime(2020, 7, 11, 0, 0), datetime.datetime(2020, 7, 10, 0, 0), datetime.datetime(2020, 7, 12, 0, 0), datetime.datetime(2020, 7, 13, 0, 0), datetime.datetime(2020, 7, 14, 0, 0), datetime.datetime(2020, 7, 15, 0, 0), datetime.datetime(2020, 7, 16, 0, 0), datetime.datetime(2020, 7, 17, 0, 0), datetime.datetime(2020, 7, 20, 0, 0), datetime.datetime(2020, 7, 19, 0, 0), datetime.datetime(2020, 7, 18, 0, 0), datetime.datetime(2020, 7, 21, 0, 0), datetime.datetime(2020, 7, 22, 0, 0), datetime.datetime(2020, 7, 23, 0, 0), datetime.datetime(2020, 7, 24, 0, 0), datetime.datetime(2020, 7, 26, 0, 0), datetime.datetime(2020, 7, 25, 0, 0), datetime.datetime(2020, 7, 29, 0, 0), datetime.datetime(2020, 6, 2, 0, 0), datetime.datetime(2020, 6, 1, 0, 0), datetime.datetime(2020, 5, 31, 0, 0), datetime.datetime(2020, 6, 3, 0, 0), datetime.datetime(2020, 6, 4, 0, 0), datetime.datetime(2020, 6, 5, 0, 0), datetime.datetime(2020, 6, 6, 0, 0), datetime.datetime(2020, 6, 7, 0, 0), datetime.datetime(2020, 6, 9, 0, 0), datetime.datetime(2020, 6, 8, 0, 0), datetime.datetime(2020, 6, 11, 0, 0), datetime.datetime(2020, 6, 10, 0, 0), datetime.datetime(2020, 6, 13, 0, 0), datetime.datetime(2020, 6, 12, 0, 0), datetime.datetime(2020, 6, 15, 0, 0), datetime.datetime(2020, 6, 14, 0, 0), datetime.datetime(2020, 6, 16, 0, 0), datetime.datetime(2020, 6, 17, 0, 0), datetime.datetime(2020, 6, 18, 0, 0), datetime.datetime(2020, 6, 19, 0, 0), datetime.datetime(2020, 6, 20, 0, 0), datetime.datetime(2020, 6, 21, 0, 0), datetime.datetime(2020, 6, 22, 0, 0), datetime.datetime(2020, 6, 23, 0, 0), datetime.datetime(2020, 6, 27, 0, 0), datetime.datetime(2020, 6, 26, 0, 0), datetime.datetime(2020, 6, 24, 0, 0), datetime.datetime(2020, 5, 27, 0, 0), datetime.datetime(2020, 5, 26, 0, 0), datetime.datetime(2020, 5, 25, 0, 0), datetime.datetime(2020, 5, 24, 0, 0), datetime.datetime(2020, 5, 23, 0, 0), datetime.datetime(2020, 5, 22, 0, 0), datetime.datetime(2020, 5, 20, 0, 0), datetime.datetime(2020, 5, 19, 0, 0), datetime.datetime(2020, 5, 21, 0, 0), datetime.datetime(2020, 5, 28, 0, 0), datetime.datetime(2020, 5, 29, 0, 0), datetime.datetime(2020, 5, 30, 0, 0), datetime.datetime(2020, 11, 10, 0, 0), datetime.datetime(2020, 11, 8, 0, 0), datetime.datetime(2020, 11, 9, 0, 0), datetime.datetime(2020, 11, 7, 0, 0), datetime.datetime(2020, 11, 5, 0, 0), datetime.datetime(2020, 11, 6, 0, 0), datetime.datetime(2020, 11, 11, 0, 0), datetime.datetime(2020, 11, 12, 0, 0), datetime.datetime(2020, 11, 13, 0, 0), datetime.datetime(2020, 11, 16, 0, 0), datetime.datetime(2020, 11, 15, 0, 0), datetime.datetime(2020, 11, 14, 0, 0), datetime.datetime(2020, 11, 17, 0, 0), datetime.datetime(2020, 11, 18, 0, 0), datetime.datetime(2020, 11, 19, 0, 0), datetime.datetime(2020, 11, 20, 0, 0), datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020, 11, 2, 0, 0), datetime.datetime(2020, 11, 1, 0, 0), datetime.datetime(2020, 10, 31, 0, 0), datetime.datetime(2020, 10, 29, 0, 0), datetime.datetime(2020, 10, 30, 0, 0), datetime.datetime(2020, 10, 26, 0, 0), datetime.datetime(2020, 10, 28, 0, 0), datetime.datetime(2020, 10, 27, 0, 0), datetime.datetime(2020, 10, 25, 0, 0), datetime.datetime(2020, 11, 4, 0, 0), datetime.datetime(2020, 11, 24, 0, 0), datetime.datetime(2020, 11, 23, 0, 0), datetime.datetime(2020, 11, 22, 0, 0), datetime.datetime(2020, 11, 21, 0, 0), datetime.datetime(2020, 11, 25, 0, 0), datetime.datetime(2020, 11, 26, 0, 0), datetime.datetime(2020, 10, 1, 0, 0), datetime.datetime(2020, 9, 30, 0, 0), datetime.datetime(2020, 9, 29, 0, 0), datetime.datetime(2020, 9, 28, 0, 0), datetime.datetime(2020, 10, 2, 0, 0), datetime.datetime(2020, 10, 3, 0, 0), datetime.datetime(2020, 10, 5, 0, 0), datetime.datetime(2020, 10, 4, 0, 0), datetime.datetime(2020, 10, 6, 0, 0), datetime.datetime(2020, 10, 7, 0, 0), datetime.datetime(2020, 10, 9, 0, 0), datetime.datetime(2020, 10, 8, 0, 0), datetime.datetime(2020, 10, 10, 0, 0), datetime.datetime(2020, 10, 13, 0, 0), datetime.datetime(2020, 10, 12, 0, 0), datetime.datetime(2020, 10, 11, 0, 0), datetime.datetime(2020, 10, 15, 0, 0), datetime.datetime(2020, 10, 14, 0, 0), datetime.datetime(2020, 10, 19, 0, 0), datetime.datetime(2020, 10, 18, 0, 0), datetime.datetime(2020, 10, 17, 0, 0), datetime.datetime(2020, 10, 16, 0, 0), datetime.datetime(2020, 10, 20, 0, 0), datetime.datetime(2020, 10, 21, 0, 0), datetime.datetime(2020, 10, 22, 0, 0), datetime.datetime(2020, 10, 23, 0, 0), datetime.datetime(2020, 10, 24, 0, 0), datetime.datetime(2020, 9, 2, 0, 0), datetime.datetime(2020, 9, 3, 0, 0), datetime.datetime(2020, 9, 4, 0, 0), datetime.datetime(2020, 9, 7, 0, 0), datetime.datetime(2020, 9, 6, 0, 0), datetime.datetime(2020, 9, 5, 0, 0), datetime.datetime(2020, 9, 8, 0, 0), datetime.datetime(2020, 9, 10, 0, 0), datetime.datetime(2020, 9, 9, 0, 0), datetime.datetime(2020, 9, 11, 0, 0), datetime.datetime(2020, 9, 14, 0, 0), datetime.datetime(2020, 9, 13, 0, 0), datetime.datetime(2020, 9, 12, 0, 0), datetime.datetime(2020, 9, 15, 0, 0), datetime.datetime(2020, 9, 16, 0, 0), datetime.datetime(2020, 9, 17, 0, 0), datetime.datetime(2020, 9, 18, 0, 0), datetime.datetime(2020, 9, 19, 0, 0), datetime.datetime(2020, 9, 21, 0, 0), datetime.datetime(2020, 9, 20, 0, 0), datetime.datetime(2020, 9, 22, 0, 0), datetime.datetime(2020, 9, 23, 0, 0), datetime.datetime(2020, 9, 24, 0, 0), datetime.datetime(2020, 9, 25, 0, 0), datetime.datetime(2020, 9, 26, 0, 0), datetime.datetime(2020, 9, 27, 0, 0)])
pro_anti = ("anti", "pro")
start_over = False
find_largest = False
text = list()
if start_over:
    for date in dates:
        for ele in pro_anti:
            infile = open("C:\\Users\\aarts\\PycharmProjects\\AntiVax\\Tweets\\TweetsPerDay\\" + str(date).split(" ")[0] +
                          "Tweets_" + ele + ".txt", "r", encoding="utf8")
            for line in infile.read().split("\n"):
                try:
                    tweet = json.loads(line)
                    if "retweeted_status" in tweet.keys():
                        text.append(tweet["retweeted_status"]["full_text"])
                    else:
                        text.append(tweet["full_text"])
                except(JSONDecodeError):
                    print(line)
            infile.close()
    tf_idf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    matrix = tf_idf.fit_transform(text)
    text.clear()
    svd = TruncatedSVD(n_components=100)
    svd.fit_transform(matrix)

    dump(tf_idf, "tf_idf_all_days.joblib")
    dump(svd, "svd_all_days.joblib")
else:
    tf_idf = load("tf_idf_all_days.joblib")
    svd = load("svd_all_days.joblib")

if find_largest:
    data = dict()

    for date in dates:
        text = list()
        data[date] = dict()
        for ele in pro_anti:
            infile = open("C:\\Users\\aarts\\PycharmProjects\\AntiVax\\Tweets\\TweetsPerDay\\" + str(date).split(" ")[0] +
                          "Tweets_" + ele + ".txt", "r", encoding="utf8")
            for line in infile.read().split("\n"):
                try:
                    tweet = json.loads(line)
                    if "retweeted_status" in tweet.keys():
                        text.append(tweet["retweeted_status"]["full_text"])
                    else:
                        text.append(tweet["full_text"])
                except(JSONDecodeError):
                    print(line)

            data[date][ele] = len(text)

    max_anti = 0
    date_anti = None
    max_pro = 0
    date_pro = None

    for date in data:
        if data[date]["pro"] > max_pro:
            max_pro = data[date]["pro"]
            date_pro = date
        if data[date]["anti"] > max_anti:
            max_anti = data[date]["anti"]
            date_anti = date

    print("Pro")
    print(date_pro)
    print(max_pro)
    print("Anti")
    print(date_anti)
    print(max_anti)

string = "2020-09-03Tweets_"
data = dict()
anti_lst = list()
pro_lst = list()

clusters = dict()

other_anti_list = list()
other_pro_list = list()

for ele in ("pro", "anti"):
    infile = open("C:\\Users\\aarts\\PycharmProjects\\AntiVax\\Tweets\\TweetsPerDay\\" + string + ele + ".txt", "r")
    data[ele] = list()
    for line in infile.read().split("\n"):
        try:
            tweet = json.loads(line)
            if "retweeted_status" in tweet.keys():
                data[ele].append(tweet["retweeted_status"]["full_text"])
            else:
                data[ele].append(tweet["full_text"])
        except(JSONDecodeError):
            print(line)

    matrix = tf_idf.transform(data[ele])
    matrix = svd.transform(matrix)
    clusters[ele] = dict()
    dbscan = DBSCAN(eps=0.25, min_samples=3)
    results = dbscan.fit_predict(matrix)
    my_dict = dict()
    for num in range(len(results)):
        if results[num] in my_dict:
            my_dict[results[num]].append(num)
        else:
            my_dict[results[num]] = list()
            my_dict[results[num]].append(num)
    print(ele)

    for cluster in my_dict:
        clusters[ele][cluster] = list()
        for element in my_dict[cluster]:
            clusters[ele][cluster].append(matrix[element])

    for point in my_dict[0]:
        if ele == "pro":
            pro_lst.append(matrix[point][0:3])
            other_pro_list.append(matrix[point])
        else:
            anti_lst.append(matrix[point][0:3])
            other_anti_list.append(matrix[point])

colors = ("b", "g", "r", "c", "m", "y", "k")
fig = plt.figure()



ax = plt.axes(projection="3d")

anti_hull = ConvexHull(anti_lst)
print(anti_hull.vertices)
print(anti_hull.simplices)
pro_hull = ConvexHull(pro_lst)
print(pro_hull)

anti_points = list()
pro_points = list()

for num in pro_hull.vertices:
    pro_points.append(pro_lst[num])
for num in anti_hull.vertices:
    anti_points.append(anti_lst[num])


# vertices_pro = [list(zip(xs_pro,ys_pro,zs_pro))]
# vertices_anti = [list(zip(xs_anti,ys_anti, zs_anti))]

# poly_pro = Poly3DCollection(vertices_pro, alpha=0.8)
# poly_anti = Poly3DCollection(vertices_anti, alpha=0.8)



# ax.add_collection(poly_pro)
# ax.add_collection(poly_anti)

# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# ax.set_zlim(-1,1)

# ax.scatter(xs_anti, ys_anti, zs_anti)
# ax.scatter(xs_pro, ys_pro, zs_pro)

ax.plot(np.asarray(anti_points).T[0], np.asarray(anti_points).T[1], np.asarray(anti_points).T[2], "ro", label="anti")
ax.plot(np.asarray(pro_points).T[0], np.asarray(pro_points).T[1], np.asarray(pro_points).T[2], "bo", label="pro")

for s in anti_hull.simplices:
    lst = list()
    lst.append(anti_lst[s[0]])
    lst.append(anti_lst[s[1]])
    lst.append(anti_lst[s[2]])
    lst.append(anti_lst[s[0]])

    ax.plot(np.asarray(lst).T[0], np.asarray(lst).T[1], np.asarray(lst).T[2], "r-", linewidth=0.5)

for s in pro_hull.simplices:
    lst = list()
    lst.append(pro_lst[s[0]])
    lst.append(pro_lst[s[1]])
    lst.append(pro_lst[s[2]])
    lst.append(pro_lst[s[0]])

    ax.plot(np.asarray(lst).T[0], np.asarray(lst).T[1], np.asarray(lst).T[2], "b-", linewidth=0.5)

# ax.plot_trisurf(xs_anti, ys_anti, zs_anti)
# ax.plot_trisurf(xs_pro, ys_pro, zs_pro)
ax.legend()
plt.show()


for anti_pro in clusters:
    print(anti_pro)
    ax = plt.axes(projection="3d")
    for cluster in clusters[anti_pro]:
        ax.plot(np.asarray(clusters[anti_pro][cluster]).T[0], np.asarray(clusters[anti_pro][cluster]).T[1],
                np.asarray(clusters[anti_pro][cluster]).T[2], colors[cluster] + "o", label=str(cluster))
    ax.legend()
    plt.show()