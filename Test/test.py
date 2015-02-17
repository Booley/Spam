import email_process as ep
import pickle
from sklearn.naive_bayes import MultinomialNB

# read bag of words
bag = ep.read_bagofwords_dat("test_emails_bag_of_words_0.dat", 5000)

labels = [line for line in open("test_emails_classes_0.txt", "r")]

# read classifier. Remember to copy file from Train folder!!!
savedFile = open("nb.cl","r")
clf = pickle.load(savedFile)

# begin testing
print clf.score(bag,labels)