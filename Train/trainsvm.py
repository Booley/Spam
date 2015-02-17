import email_process as ep
import pickle
from sklearn import svm

# read bag of words
print "Reading in bag of words"
bag = ep.read_bagofwords_dat("train_emails_bag_of_words_200.dat")


# train = []
# # read labels
# classes_fp = open("train_emails_classes_200.txt", "r")
# train = [(bag[i], label) for i, label in enumerate(classes_fp)]

labels = [line for line in open("train_emails_classes_200.txt", "r")]

# create decision tree
print "Training classifier"
clf = svm.SVC()
clf.fit(bag, labels)

# save classifier. Remember to move file to Test folder!!!
print "Saving classifier"
saveFile = open("svm.cl", "w")
pickle.dump(clf, saveFile)
saveFile.close()

