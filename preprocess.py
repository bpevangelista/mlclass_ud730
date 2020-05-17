import imageio, os
import numpy as np
import pickle # serialization

kImageWidth = 28
kImageHeight = 28
kImageSize = kImageWidth * kImageHeight
kLargeFolder = 'notMNIST_large'
kSmallFolder = 'notMNIST_small'


def loadAllImages(mainSet, label, loadPath):
    imageNames = os.listdir(loadPath)
    imageNamesCount = len(imageNames)
    dataset = np.ndarray(shape=(imageNamesCount, kImageSize), dtype=np.float32)

    print('Processing {0}, with {1} images...'.format(loadPath, imageNamesCount))
    imageCount = 0
    for imageName in imageNames:
        imagePath = os.path.join(loadPath, imageName)
        try:
            imageData = imageio.imread(imagePath).flatten()
            imageData = (imageData.astype(float) - 128.0) / 255.0 # normalize [-1, 1]
            #print(imageData)
            if imageData.shape[0] != kImageSize:
                print('  Skipping invalid image size: {0}'.format(imagePath))
                continue
            dataset[imageCount, :] = imageData
            imageCount += 1
        except (IOError, ValueError) as err:
            print('  Skipping invalid image data: {0}'.format(imagePath))
    
    # Possibly reduce dataset size, due to invalid image data
    dataset = dataset[0:imageCount, :]
    print('  Dataset: {0}, Mean: {1}, StdDev: {2}'.format(
        dataset.shape, np.mean(dataset), np.std(dataset)))

    #dumpPath = loadPath + '.pickle'
    #print('  Writing: {0}'.format(dumpPath))
    #with open(dumpPath, 'wb') as dumpFile:
    #   pickle.dump(dataset, dumpFile, pickle.HIGHEST_PROTOCOL)

    labels = np.full(len(dataset), label, dtype=np.int32)
    if label in mainSet['dataset']:
        mainSet['dataset'][label] = np.append(mainSet['dataset'][label], dataset, axis=0)
        mainSet['labels'][label] = np.append(mainSet['labels'][label], labels, axis=0)
    else:
        mainSet['dataset'][label] = dataset
        mainSet['labels'][label] = labels

    print('  Mainset: {0}, Mean: {1}, StdDev: {2}'.format(
        mainSet['dataset'][label].shape, 
        np.mean(mainSet['dataset'][label]), 
        np.std(mainSet['dataset'][label])))
    print('  Labels: {0}, #{1}'.format(
        mainSet['labels'][label].shape, label));

def loadAllImagesWithFolderIndexAsLabel(blob, mainFolder):
    imageFolders = sorted(os.listdir(mainFolder))
    for label, imageFolder in enumerate(imageFolders):
        loadAllImages(blob, label, os.path.join(mainFolder, imageFolder));

def balanceKeys(blob):
    firstKey = next(iter(blob['dataset']))
    minCount = len(blob['dataset'][firstKey])
    print('Setting datasets size count: {0}'.format(minCount))

    for key in blob['dataset']:
        count = len(blob['dataset'][key])
        minCount = count if count < minCount else minCount
    for key in blob['dataset']:
        blob['dataset'][key] = blob['dataset'][key][0:minCount]
        blob['labels'][key] = blob['labels'][key][0:minCount]


def randomize(blob):
    for key in blob['dataset']:
        permutation = np.random.permutation(len(blob['dataset'][key]))
        blob['dataset'][key] = blob['dataset'][key][permutation, :]
        blob['labels'][key] = blob['labels'][key][permutation]

def main():
    mainSet = {
        'dataset': {},
        'labels': {},
    }

    loadAllImagesWithFolderIndexAsLabel(mainSet, kSmallFolder);
    loadAllImagesWithFolderIndexAsLabel(mainSet, kLargeFolder);
    randomize(mainSet);
    balanceKeys(mainSet);

    with open('mainset.pickle', 'wb') as outputFile:
        pickle.dump(mainSet, outputFile, pickle.HIGHEST_PROTOCOL)
        outputFile.close()

main()