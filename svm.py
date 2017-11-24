import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

# import data
labeled_images = pd.read_csv('data/train.csv')

# create training data and test data
images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# view an image of a digit
i = 1
img = train_images.iloc[i].as_matrix()
img = img.reshape((28, 28))
my_plot = plt.imshow(img, cmap='gray')
my_plot = plt.title(train_labels.iloc[i, 0])
plt.show(my_plot)

# histogram of pixel values
my_hist = plt.hist(train_images.iloc[i])
plt.show(my_hist)

# Use Vector classifier to train and score model
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images, test_labels)

# Improve model by making any pixel with a value a 1 and no value as 0
test_images[test_images > 0] = 1
train_images[train_images > 0] = 1

# View binary image
img = train_images.iloc[i].as_matrix().reshape((28, 28))
my_plot = plt.imshow(img, cmap='binary')
my_plot = plt.title(train_labels.iloc[i])

# Retrain using binary data
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images, test_labels)

# Predict values of competition data
test_data = pd.read_csv('data/test.csv')
test_data[test_data > 0] = 1
results = clf.predict(test_data[0:])
results

# Save results
df = pd.DataFrame(results)
df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('data/results.csv', header=True)
# blurg
