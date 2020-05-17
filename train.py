import imageio, os
import numpy as np
import pickle # serialization

from PIL import Image # Python Image library
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

kLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
kImageWidth = 28
kImageHeight = 28
kMainSetPath = 'mainset.pickle'
kSplitSetPath = 'mainset_split.pickle'

train_size = 400000
valid_size = 40000
test_size = 40000
#train_size = 200000
#valid_size = 10000
#test_size = 10000

def splitDataset(inDataPath, outDataPath):
    mainset = {}
    with open(inDataPath, 'rb') as mainsetFile:
        mainset = pickle.load(mainsetFile)

    # Info
    print('Mainset:')
    for key in mainset['dataset']:
        print('  [{0}] Images: {1}, Labels: {2}'.format(
            kLabels[key], len(mainset['dataset'][key]), len(mainset['labels'][key])))

    mergedDataset = None
    mergedLabels = None
    for key in mainset['dataset']:
        if mergedDataset is None:
            mergedDataset = mainset['dataset'][key]
            mergedLabels = mainset['labels'][key]
        else:
            mergedDataset = np.append(mergedDataset, mainset['dataset'][key], axis=0)
            mergedLabels = np.append(mergedLabels, mainset['labels'][key])

    # Randomize
    permutation = np.random.permutation(mergedDataset.shape[0])
    mergedDataset = mergedDataset[permutation, :]
    mergedLabels = mergedLabels[permutation]

    # Make sure we have enough data
    required_size = train_size + valid_size + test_size
    if len(mergedDataset) < required_size:
        print('Not enough samples. Required {0}, found {1}'.format(required_size, len(mergedDataset)))
        return

    startIndex = 0
    trainBlob = {}
    trainBlob['train_dataset'] = mergedDataset[startIndex:startIndex+train_size]
    trainBlob['train_labels'] = mergedLabels[startIndex:startIndex+train_size]
    startIndex += train_size
    trainBlob['validate_dataset'] = mergedDataset[startIndex:startIndex+valid_size]
    trainBlob['validate_labels'] = mergedLabels[startIndex:startIndex+valid_size]
    startIndex += valid_size
    trainBlob['test_dataset'] = mergedDataset[startIndex:startIndex+test_size]
    trainBlob['test_labels'] = mergedLabels[startIndex:startIndex+test_size]
    startIndex += test_size
    #print(trainBlob)

    # Info
    print('Train     Images: {0}, Labels:{1}'.format(
        len(trainBlob['train_dataset']), len(trainBlob['train_labels'])))
    print('Validate  Images: {0}, Labels:{1}'.format(
        len(trainBlob['validate_dataset']), len(trainBlob['validate_labels'])))
    print('Test      Images: {0}, Labels:{1}'.format(
        len(trainBlob['test_dataset']), len(trainBlob['test_labels'])))
    print('Remaining Images: {0}, Labels:{1}'.format(
        len(mergedDataset) - startIndex, len(mergedLabels) - startIndex))

    with open(outDataPath, 'wb') as outputFile:
        pickle.dump(trainBlob, outputFile, pickle.HIGHEST_PROTOCOL)
        outputFile.close()


def randomVerify(dataBlob):
    for i in range(10):
        imageIndex = np.random.randint(0, len(dataBlob['train_dataset']));
        imageData = dataBlob['train_dataset'][imageIndex]
        imageData = imageData * 255.0 + 128.0
        arrImage = Image.fromarray(imageData.reshape(kImageWidth, kImageHeight), "F")
        
        imageLabel = dataBlob['train_labels'][imageIndex]
        arrImage.show(title=kLabels[imageLabel])
        print(kLabels[imageLabel])


def main():
    if not os.path.isfile(kSplitSetPath):
        print('Splitting data...')
        splitDataset(kMainSetPath, kSplitSetPath)

    print('Loading data...')
    with open(kSplitSetPath, 'rb') as inputFile:
        trainBlob = pickle.load(inputFile)
        
        # Debug
        #randomVerify(trainBlob)

        print('Training...')
        #model = linear_model.LinearRegression()
        model = linear_model.LogisticRegression(max_iter=200)
        model.fit(trainBlob['train_dataset'], trainBlob['train_labels'])

        print('Sklearn mean_squared_error: %.2f' % mean_squared_error(trainBlob['validate_labels'], 
            model.predict(trainBlob['validate_dataset'])))

        print('Sklearn R2_score: %.2f' % r2_score(trainBlob['validate_labels'], 
            model.predict(trainBlob['validate_dataset'])))

main()