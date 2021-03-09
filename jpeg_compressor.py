import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from itertools import chain
from collections import Counter
from os import path

# High Quality Quantization table (lower degree of compression)
quantizationTableHigh = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]])

# Lower Quality Quantization table (higher degree of compression)
quantizationTableLow = np.array([[10, 10, 76, 255, 255, 255, 255, 255],
                                 [85, 112, 255, 255, 255, 255, 255, 255],
                                 [151, 255, 255, 255, 255, 255, 255, 255],
                                 [255, 255, 255, 255, 255, 255, 255, 255],
                                 [255, 255, 255, 255, 255, 255, 255, 255],
                                 [255, 255, 255, 255, 255, 255, 255, 255],
                                 [255, 255, 255, 255, 255, 255, 255, 255],
                                 [255, 255, 255, 255, 255, 255, 255, 255]])


# zig zag pattern used for run length coding
zigzagPattern = [
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63]


class JPEG:
    def __init__(self, imgName, blockSize=[8, 8], subSampling=[4, 2, 2], compressionQuality = "high"):
        self.img = mpimg.imread(imgName)
        self.originalShape = [self.img.shape[0], self.img.shape[1]]
        self.blockSize = blockSize
        self.subSampling = subSampling
        self.paddedShape = None
        self.paddedResolution = None
        self.huffmanCodeList = None
        self.huffmanRunLength = None
        self.start_time = time.time()
        if compressionQuality == "high":
            self.quantizationtable = quantizationTableHigh
        else:
            self.quantizationtable = quantizationTableLow

    def compress_img(self):
        show_img(self.img)
        show_img(self.img, True)
        # pad image if pixel length or width %8 != 0
        self.img = pad_image_with_zeros(self.img)
        self.paddedShape = [int(self.img.shape[0] / 8), int(self.img.shape[1] / 8)]
        start_time = time.time()
        print("Padding took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # convert image from rgb to yCbCr
        self.img = rgb2ycbcr(self.img)
        print("rgb to yCbCr took --- %s seconds ---" % (time.time() - start_time))
        show_img(self.img, True)

        chroma_subsampling(self.img, self.subSampling)
        print("subsampling took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        show_img(self.img, True)

        # slice images in 8x8 blocks
        self.img = slice_image(self.img, self.blockSize)
        print("Slicing took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # dct transform blocks of image
        self.img = dct_tranform_img_blocks(self.img, False)
        print("DCT took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # apply quantisation table to blocks of image
        self.img = manage_quantization_with_table(self.img, self.quantizationtable)

        # zig zag date of each block in image
        self.img = zig_zag_blocks(self.img)

        # apply huffman coding an runlength coding to each block of the image
        self.huffmanCodeList = calc_huffman_tree_of_blocks(self.img)
        print("Huffman took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        self.huffmanRunLength = calc_huffman_code_of_blocks(self.img, self.huffmanCodeList)
        print("run length took --- %s seconds ---" % (time.time() - start_time))

    def decompress_img(self):
        # decode huffman code of the image blocks
        self.img = decode_huffman(self.huffmanCodeList, self.huffmanRunLength, self.paddedShape)
        start_time = time.time()
        print("decoded huffman took --- %s seconds ---" % (time.time() - start_time))

        # de quantize the image by once again applying the quantization table
        self.img = manage_quantization_with_table(self.img, self.quantizationtable, True)
        print("qunatisation reverse took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # inverse DCT transform of each block of the image
        self.img = dct_tranform_img_blocks(self.img, True)
        print("idct took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # put data back into a nparray to display as image
        self.img = put_data_back_nparray(self.img, [i * 8 for i in self.paddedShape])
        print("reshaping array took --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # convert yCbCr color space to rgb
        self.img[:, :, [0, 1, 2]] += 128
        self.img = ycbcr2rgb(self.img)
        print("yCbCr to RGB took --- %s seconds ---" % (time.time() - start_time))

        self.img = remove_image_padding(self.img, self.originalShape)

        show_img(self.img)


# Function to display current image, if components true, displays values of each pixel separately
def show_img(img, components=False):
    if components:
        f, axarr = plt.subplots(2, 4)

        firstCompImg = img.copy()
        firstCompImg[:, :, 1] = 0
        firstCompImg[:, :, 2] = 0
        axarr[0, 0].imshow(firstCompImg)

        secondCompImg = img.copy()
        secondCompImg[:, :, 0] = 0
        secondCompImg[:, :, 2] = 0
        axarr[0, 1].imshow(secondCompImg)

        thirdCompImg = img.copy()
        thirdCompImg[:, :, 0] = 0
        thirdCompImg[:, :, 1] = 0
        axarr[0, 2].imshow(thirdCompImg)

        axarr[0, 3].imshow(img)

        axarr[1, 0].imshow(firstCompImg[:, :, 0], cmap='gray')

        axarr[1, 1].imshow(secondCompImg[:, :, 1], cmap='gray')

        axarr[1, 2].imshow(thirdCompImg[:, :, 2], cmap='gray')

        axarr[1, 3].imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


# converts image from rgb to yCbCr color space by utilizing the conversion matrix from
# https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-rdprfx/b550d1b5-f7d9-4a0c-9141-b3dca9d7f525?redirectedfrom=MSDN
# the y component is shifted by 128 to fall into the [-128.0, 127.0] range.
def rgb2ycbcr(im):
    xform = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.500], [0.500, -0.418688, -0.081312]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


# converts image from yCbCr to rgb color space by utilizing the conversion matrix from
# https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-rdprfx/2e1618ed-60d6-4a64-aa5d-0608884861bb
# the y component is shifted by 128 so that it falls in the [0.0, 255.0] range.
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


# subsampling cB and cR components of the image, combines pixel to subpixel and calcualte
# the average of these values
def chroma_subsampling(inputImg, subSamplingMethod=[4, 2, 2]):
    for row in range(0, len(inputImg), 2):
        for col in range(0, len(inputImg[0]), 4):
            # subsample Cb, Cr component
            sampleBlocks = np.stack((inputImg[row][col:col + 4], inputImg[row + 1][col:col + 4]))
            summedBlockCb = []
            summedBlockCr = []
            # [4:1:1]
            if subSamplingMethod == [4, 1, 1]:
                summedBlockCb.append(
                    sum((sampleBlocks[0][0][1], sampleBlocks[0][1][1], sampleBlocks[0][2][1], sampleBlocks[0][3][1])))
                summedBlockCb.append(
                    sum((sampleBlocks[1][0][1], sampleBlocks[1][1][1], sampleBlocks[1][2][1], sampleBlocks[1][3][1])))

                summedBlockCr.append(
                    sum((sampleBlocks[0][0][2], sampleBlocks[0][1][2], sampleBlocks[0][2][2], sampleBlocks[0][3][2])))
                summedBlockCr.append(
                    sum((sampleBlocks[1][0][2], sampleBlocks[1][1][2], sampleBlocks[1][2][2], sampleBlocks[1][3][2])))

                for x in range(0, 4):
                    inputImg[row][col + x][1] = summedBlockCb[0] / 4
                    inputImg[row + 1][col + x][1] = summedBlockCb[1] / 4
                    inputImg[row][col + x][2] = summedBlockCr[0] / 4
                    inputImg[row + 1][col + x][2] = summedBlockCr[1] / 4
            # [4:2:0]
            elif subSamplingMethod == [4, 2, 0]:
                summedBlockCb.append(
                    sum((sampleBlocks[0][0][1], sampleBlocks[0][1][1], sampleBlocks[1][0][1], sampleBlocks[1][1][1])))
                summedBlockCb.append(
                    sum((sampleBlocks[0][2][1], sampleBlocks[0][3][1], sampleBlocks[1][2][1], sampleBlocks[1][3][1])))

                summedBlockCr.append(
                    sum((sampleBlocks[0][0][2], sampleBlocks[0][1][2], sampleBlocks[1][0][2], sampleBlocks[1][1][2])))
                summedBlockCr.append(
                    sum((sampleBlocks[0][2][2], sampleBlocks[0][3][2], sampleBlocks[1][2][2], sampleBlocks[1][3][2])))

                for y in range(0, 4):
                    for x in range(0, 2):
                        inputImg[row + x][col + y][1] = summedBlockCb[x] / 4
                        inputImg[row + x][col + y][2] = summedBlockCr[x] / 4
            # [4:2:2]
            elif subSamplingMethod == [4, 2, 2]:
                summedBlockCb.append(sum((sampleBlocks[0][0][1], sampleBlocks[0][1][1])))
                summedBlockCb.append(sum((sampleBlocks[0][2][1], sampleBlocks[0][3][1])))
                summedBlockCb.append(sum((sampleBlocks[1][0][1], sampleBlocks[1][1][1])))
                summedBlockCb.append(sum((sampleBlocks[1][2][1], sampleBlocks[1][3][1])))

                summedBlockCr.append(sum((sampleBlocks[0][0][2], sampleBlocks[0][1][2])))
                summedBlockCr.append(sum((sampleBlocks[0][2][2], sampleBlocks[0][3][2])))
                summedBlockCr.append(sum((sampleBlocks[1][0][2], sampleBlocks[1][1][2])))
                summedBlockCr.append(sum((sampleBlocks[1][2][2], sampleBlocks[1][3][2])))

                setIndex = 0
                for x in range(0, 2):
                    for y in range(0, 4):
                        if x == 0 and y == 2:
                            setIndex += 1
                        elif x == 1 and y == 0:
                            setIndex += 1
                        elif x == 1 and y == 2:
                            setIndex += 1
                        inputImg[row + x][col + y][1] = summedBlockCb[setIndex] / 2
                        inputImg[row + x][col + y][2] = summedBlockCr[setIndex] / 2

            # [4:4:0]
            elif subSamplingMethod == [4, 4, 0]:
                summedBlockCb.append(sum((sampleBlocks[0][0][1], sampleBlocks[1][0][1])))
                summedBlockCb.append(sum((sampleBlocks[0][1][1], sampleBlocks[1][1][1])))
                summedBlockCb.append(sum((sampleBlocks[0][2][1], sampleBlocks[1][2][1])))
                summedBlockCb.append(sum((sampleBlocks[0][3][1], sampleBlocks[1][3][1])))

                summedBlockCr.append(sum((sampleBlocks[0][0][2], sampleBlocks[1][0][2])))
                summedBlockCr.append(sum((sampleBlocks[0][1][2], sampleBlocks[1][1][2])))
                summedBlockCr.append(sum((sampleBlocks[0][2][2], sampleBlocks[1][2][2])))
                summedBlockCr.append(sum((sampleBlocks[0][3][2], sampleBlocks[1][3][2])))

                setIndex = 0
                for y in range(0, 4):
                    for x in range(0, 2):
                        inputImg[row + x][col + y][1] = summedBlockCb[y] / 2
                        inputImg[row + x][col + y][2] = summedBlockCr[y] / 2
            # no subsampling
            else:
                return


# slices the image in 8x8 (64 pixel) blocks, these values are centered around zero by substituting 128 from each
def slice_image(inputImg, inputBlockSize):
    rowCounter = inputImg.shape[0]
    columnCounter = inputImg.shape[1]
    slicedBlock = [[None for _ in range(int(columnCounter / inputBlockSize[0]))] for _ in
                   range(int(rowCounter / inputBlockSize[1]))]
    sliceBlockCounterRow = -1
    for row in range(0, rowCounter, inputBlockSize[0]):
        sliceBlockCounterRow += 1
        sliceBlockCounterCol = -1
        for col in range(0, columnCounter, inputBlockSize[1]):
            sliceBlockCounterCol += 1
            slicedBlock[sliceBlockCounterRow][sliceBlockCounterCol] = np.float32(
                inputImg[row:row + inputBlockSize[0], col:col + inputBlockSize[1]] - 128.0)
    return slicedBlock


# 2d discrete cosine transform on a 8x8 pixel block.
# Image pixels represented by different cosine waves, high frequencies are eliminated
# values in the upper left corner are dominant while the rest of the matrix becomes almost zero
# u = horizontal spatial frequency, v = vertical spatial frequency,
def dct2(inputMatrix):
    outputMatrix = inputMatrix.copy()
    for u in range(0, 8):
        for v in range(0, 8):
            if u == 0:
                Cu = 1 / np.sqrt(8)
            else:
                Cu = np.sqrt(2 / 8)
            if v == 0:
                Cv = 1 / np.sqrt(8)
            else:
                Cv = np.sqrt(2 / 8)
            sumVal = 0
            for x in range(0, 8):
                for y in range(0, 8):
                    sumVal += inputMatrix[x][y] * np.cos(((2.0 * x + 1) * u * np.pi) / 16.0) * np.cos(
                        ((2.0 * y + 1) * v * np.pi) / 16.0)
            outputMatrix[u][v] = Cu * Cv * sumVal
    return outputMatrix


# the inverse discrete cosine transform, converts frequency to intensity data
def idct2(inputMatrix, Cu=1, Cv=1):
    outputMatrix = inputMatrix.copy()
    for x in range(0, 8):
        for y in range(0, 8):
            sumVal = 0
            for u in range(0, 8):
                for v in range(0, 8):
                    if u == 0:
                        Cu = 1 / np.sqrt(8)
                    else:
                        Cu = np.sqrt(2 / 8)
                    if v == 0:
                        Cv = 1 / np.sqrt(8)
                    else:
                        Cv = np.sqrt(2 / 8)
                    sumVal += Cu * Cv * inputMatrix[u][v] * np.cos(((2.0 * x + 1) * u * np.pi) / 16.0) * np.cos(
                        ((2.0 * y + 1) * v * np.pi) / 16.0)
            outputMatrix[x][y] = sumVal
    return outputMatrix


# for encoding, values of each block are divided by the quantisation table
# by casting the result to integer, values close to 0 become zer0
# for decoding, values of each block are multiplied by the quantisation table
def quantization_with_table(inputMatrix, quantizationTable, inverse=False):
    outputMatrix = inputMatrix.copy()
    for u in range(0, 8):
        for v in range(0, 8):
            if inverse:
                outputMatrix[u][v] = outputMatrix[u][v] * quantizationTable[u][v]
            else:
                outputMatrix[u][v] = (outputMatrix[u][v] / quantizationTable[u][v]).astype(int)
    return outputMatrix.astype(int)


# calls the quantization function for each 8x8 block of the image
def manage_quantization_with_table(inputMatrix, quantizationTable, inverse=False):
    outputMatrix = inputMatrix.copy()
    for row in range(0, len(inputMatrix)):
        for col in range(0, len(inputMatrix[0])):
            outputMatrix[row][col] = quantization_with_table(outputMatrix[row][col], quantizationTable, inverse)
    return outputMatrix


# handles dct and idct function calls dor each block of an image
def dct_tranform_img_blocks(inputMatrix, inverse):
    outputMatrix = inputMatrix.copy()
    for row in range(0, len(inputMatrix)):
        for col in range(0, len(inputMatrix[0])):
            if inverse == False:
                outputMatrix[row][col] = dct2(outputMatrix[row][col])
            else:
                outputMatrix[row][col] = idct2(outputMatrix[row][col])
    return outputMatrix


# a help function to put the image back in a numpy Array
# matplotlib needs this presentation to display it as an image
def put_data_back_nparray(npArrayImage, paddedImgShape):
    decodedMatrix = np.empty((paddedImgShape[0], paddedImgShape[1], 3)).astype(int)
    countRows = 0
    countCols = 0
    for row in range(0, len(npArrayImage)):
        for rowBlock in range(0, len(npArrayImage[0][0])):
            for col in range(0, len(npArrayImage[0])):
                for colBlock in range(0, len(npArrayImage[0][0][0])):
                    decodedMatrix[countRows][countCols] = (npArrayImage[row][col][rowBlock][colBlock])
                    countCols += 1
                    if countCols == paddedImgShape[1]:
                        countRows += 1
                        countCols = 0
    return decodedMatrix


# to result of each block is lined up in a zig zag pattern (according to the numbers in the zigZagPattern matrix)
# This results in the zero values being after one another
def zig_zag_blocks(compressedBlocks):
    zigZagImage = []
    for blockRow in range(0, len(compressedBlocks)):
        for blockCol in range(0, len(compressedBlocks[0])):
            flattenedBlock = list(chain.from_iterable(compressedBlocks[blockRow][blockCol]))
            rearragnedBlock = []
            for element in zigzagPattern:
                for yCbCrvalue in flattenedBlock[element]:
                    rearragnedBlock.append(yCbCrvalue)
            zigZagImage.append(rearragnedBlock)
    return zigZagImage


# for decoding, to reconstruct the original order of the pixel values in each block, the zig zag pattern is applied
# it reorders the values of each block
def reverse_zig_zag_blocks(decompressedBlock):
    dezigZagImage = decompressedBlock.copy()
    for idx in range(0, len(decompressedBlock)):
        dezigZagImage[zigzagPattern[idx]] = list(map(int, decompressedBlock[idx]))
    return dezigZagImage


# uses the current blocks as python dictionaries (key and value)
# the probabilities of each value in a dictionary are calculated & used for huffman encoding
def calc_huffman_tree_of_blocks(zigZagImage):
    huffManTreeList = []
    for block in zigZagImage:
        # count all unique elements in each block
        # calculate probability of unique element in block
        blockDictionary = Counter(block)
        for key in blockDictionary:
            blockDictionary[key] /= len(block)
        pBlockDictionary = {str(k): float(v) for k, v in blockDictionary.items()}
        huffmanedBlock = huffman(pBlockDictionary)
        huffManTreeList.append(huffmanedBlock)
    return huffManTreeList


# creates huffman tree for 8x8 pixel block,
# keys are added as strings to always create unique combinations of nodes as keys
# applies run length coding as well by counting same elements after one another
def calc_huffman_code_of_blocks(zigZagImage, huffManTreeList):
    huffManRunLength = []
    blockCounter = 0
    counter = 0
    for block in zigZagImage:
        sameElementCounter = 1
        elementList = []
        for idx in range(0, len(block)):
            if idx != len(block) - 1:
                if block[idx] == block[idx + 1]:
                    sameElementCounter += 1
                else:
                    keyToGet = str(block[idx])
                    elementList.append((sameElementCounter, (huffManTreeList[blockCounter])[keyToGet]))
                    sameElementCounter = 1
            else:
                keyToGet = str(block[idx])
                elementList.append((sameElementCounter, (huffManTreeList[blockCounter])[keyToGet]))
        blockCounter += 1
        counter += 1
        huffManRunLength.append(elementList)
    return huffManRunLength


# creates huffman nodes by merging nodes with the lowest propabilities to new combined node
# the 'x' is inserted to differentiate between differentiate value pairs
# Recurse and construct code on new distribution, new nodes will be used to create more nodes and so on

def huffman(pBlockDictionary):
    # Base case of only two symbols, assign 0 or 1
    if len(pBlockDictionary) == 2:
        return dict(zip(pBlockDictionary.keys(), ['0', '1']))

    tempPBlockDictionary = pBlockDictionary.copy()
    val1, val2 = lowest_prob_pair(pBlockDictionary)
    popVal1, popVal2 = tempPBlockDictionary.pop(val1), tempPBlockDictionary.pop(val2)
    tempPBlockDictionary[(val1 + "x" + val2)] = popVal1 + popVal2

    combinedDictionary = huffman(tempPBlockDictionary)
    combVal1Val2 = combinedDictionary.pop((val1 + "x" + val2))
    combinedDictionary[val1], combinedDictionary[val2] = combVal1Val2 + '0', combVal1Val2 + '1'

    # print("Huffman code " + str(c))
    return combinedDictionary


# Return pair of symbols from distribution p with lowest probabilities.
def lowest_prob_pair(pBlockDictionary):
    sortedpBlockDictionary = sorted(pBlockDictionary.items(), key=lambda item: item[1])
    return sortedpBlockDictionary[0][0], sortedpBlockDictionary[1][0]


# help function to reshape list into matrix
def to_matrix(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


# uses the created huffman code (including run length), and decodes it with the created huffman tree
# sorts the values according to the zig zag pattern to recreate the original order of each block
def decode_huffman(huffManCodeList, huffManRunLength, paddedShape):
    decodedBlocks = []
    for idx in range(0, len(huffManCodeList)):
        decodedBlock = []
        swappedHuffManDict = dict([(value, key) for key, value in (huffManCodeList[idx]).items()])
        for element in huffManRunLength[idx]:
            for count in range(0, element[0]):
                decodedBlock.append(swappedHuffManDict.get(element[1]))
        # reverse zig zag pattern
        deZigZagDecodeBlock = reverse_zig_zag_blocks(to_matrix(decodedBlock, 3))
        # transform each block to 2d list before appending
        decodedBlocks.append(np.array(to_matrix(deZigZagDecodeBlock, 8)))
    sortedDecodedBlocks = [[None for i in range(paddedShape[1])] for j in range(paddedShape[0])]
    blockCounter = 0
    for row in range(0, paddedShape[0]):
        for col in range(0, paddedShape[1]):
            # print("row " +str(row) + " col " +str(col))
            sortedDecodedBlocks[row][col] = decodedBlocks[blockCounter]
            blockCounter += 1
    return sortedDecodedBlocks


# if the image width or height is not a multiple of 8, the image has to be padded to become a multiple of zero
# this is done to get 8x8 pixel blocks
def pad_image_with_zeros(img):
    blockSize = 8
    rowPad = blockSize - (np.mod(img.shape[0], blockSize))
    colPad = blockSize - (np.mod(img.shape[1], blockSize))
    if rowPad == blockSize:
        rowPad = 0
    if colPad == blockSize:
        colPad = 0
    return np.pad(img, ((0, rowPad), (0, colPad), (0, 0)), 'constant')


# removes padding from image to get it back to its original shape
def remove_image_padding(img, originalShape):
    return img[:originalShape[0], :originalShape[1], :]


# simple function for user input
def ask_input_file():
    inputImg = input("Enter Image name: ")
    if path.exists(inputImg) and inputImg.endswith('.jpg'):
        return inputImg
    print("File not found or in wrong format")
    return ask_input_file()


# simple function for user input
def ask_subsampling():
    subSamplingOptions = {"1": [4, 4, 4], "2": [4, 2, 2], "3": [4, 1, 1], "4": [4, 2, 0], "5": [4, 4, 0]}
    print(subSamplingOptions)
    inputSubSamp = input("Choose subsampling: ")
    if inputSubSamp in subSamplingOptions:
        return subSamplingOptions.get(inputSubSamp)
    print("Option not available")
    return ask_subsampling()


# simple function for user input
def ask_compression_quality():
    compressionOptions = {"1": "high", "2": "low"}
    print(compressionOptions)
    inputComp = input("Choose subsampling: ")
    if inputComp in compressionOptions:
        return compressionOptions.get(inputComp)
    print("Option not available")
    return ask_compression_quality()


# main function calling the compression and decompression of the image
if __name__ == '__main__':
    imgToCompress = ask_input_file()
    subSamplingMethod = ask_subsampling()
    compressionQuality = ask_compression_quality()
    firstImg = JPEG(imgToCompress, [8, 8], [4, 1, 1], compressionQuality)
    firstImg.compress_img()
    firstImg.decompress_img()
