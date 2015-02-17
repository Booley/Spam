import email_process as ep
import pickle
from sklearn.naive_bayes import MultinomialNB

# read bag of words
bag = ep.read_bagofwords_dat("train_emails_bag_of_words_200.dat")


# train = []
# # read labels
# classes_fp = open("train_emails_classes_200.txt", "r")
# train = [(bag[i], label) for i, label in enumerate(classes_fp)]

labels = [line for line in open("train_emails_classes_200.txt", "r")]

# create naive bayes classifier
clf = MultinomialNB()
clf.fit(bag, labels)

# save classifier. Remember to move file to Test folder!!!
saveFile = open("classifier", "w")
pickle.dump(clf, saveFile)
saveFile.close()