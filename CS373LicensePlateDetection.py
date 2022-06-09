import math
import sys
import queue
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# _________HelperFuncs__________
def convertIMGtoGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for height in range(image_height):
        for width in range(image_width):
            r = pixel_array_r[height][width]
            g = pixel_array_g[height][width]
            b = pixel_array_b[height][width]
            x = round(0.299 * r + 0.587 * g + 0.114 * b)
            greyscale_pixel_array[height][width] = x
    return greyscale_pixel_array

def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_value = 255
    max_value = 0
    for height in range(image_height):
        for width in range(image_width):
            x = pixel_array[height][width]
            if x > max_value:
                max_value = x
            elif x < min_value:
                min_value = x
    return (min_value, max_value)

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    min_value, max_value = computeMinAndMaxValues(pixel_array, image_width, image_height)
    if min_value == max_value:
        return result
    for height in range(image_height):
        for width in range(image_width):
            x = pixel_array[height][width]
            s = round((x - min_value) * ((255 - 0) / (max_value - min_value)))
            if s < 0:
                result[height][width] = 0
            elif s > 255:
                result[height][width] = 255
            else:
                result[height][width] = s
    return result

def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for height in range(2, image_height-2):
        top = height - 2
        bottom = height + 2
        for width in range(2, image_width-2):
            left = width - 2
            right = width + 2
            meanList = []
            for level in range(top, bottom+1):
                for length in range(left, right+1):
                    pix = pixel_array[level][length]
                    meanList.append(pix)
            mean = 0
            for num in meanList:
                mean += num
            mean = mean/len(meanList)
            variance = 0
            for m in range(len(meanList)):
                variance += pow(meanList[m] - mean, 2)
            variance = variance/len(meanList)
            result[height][width] = math.sqrt(variance)
    return result

def imageThreshholding(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for height in range(image_height):
        for width in range(image_width):
            if pixel_array[height][width] >= 150:
                result[height][width] = 255
            else:
                result[height][width] = 0
    return result

def computeDilation3x3(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for height in range(1, image_height - 1):
        top = height - 1
        bottom = height + 1
        for width in range(1, image_width - 1):
            left = width - 1
            right = width + 1
            num = 0
            for level in range(top, bottom + 1):
                for length in range(left, right + 1):
                    x = pixel_array[level][length]
                    if x > num:
                        num = x
            if num > 0:
                result[height][width] = 255
            else:
                result[height][width] = 0
    return result

def computeErosion3x3(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for height in range(1, image_height - 1):
        top = height - 1
        bottom = height + 1
        for width in range(1, image_width - 1):
            left = width - 1
            right = width + 1
            hit = 255
            for level in range(top, bottom + 1):
                for length in range(left, right + 1):
                    x = pixel_array[level][length]
                    if x < hit:
                        hit = x
            if hit < 1:
                result[height][width] = 0
            else:
                result[height][width] = 255
    return result

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    resultDict = {}
    label = 0
    seenDict = {}
    for p in range(1, image_height-1):
        for o in range(1, image_width-1):
            seenDict[p, o] = 0

    for height in range(1, image_height-1):
        for width in range(1, image_width-1):
            if pixel_array[height][width] > 0 and seenDict[height, width] == 0:
                label += 1
                resultDict[label] = 0
                seenDict[height, width] = 1
                q = queue.Queue()
                q.put((height, width))
                while q.qsize() > 0:
                    (y, x) = q.get()
                    result[y][x] = label
                    resultDict[label] += 1
                    searchArea = [1, -1, 0, 0]
                    for z in range(len(searchArea)):
                        vertSearch = y + searchArea[z]
                        horzSearch = x + searchArea[3-z]
                        if (pixel_array[vertSearch][horzSearch] > 0) and (seenDict[vertSearch, horzSearch] == 0):
                            q.put((vertSearch, horzSearch))
                            seenDict[vertSearch, horzSearch] = 1
    return result, resultDict

def computeBoundingBoxMinMax(connectedCompArray, largestComponent, image_width, image_height):
    minHeight = image_height
    minWidth = image_width
    maxHeight = 0
    maxWidth = 0
    for height in range(image_height):
        for width in range(image_width):
            if connectedCompArray[height][width] == largestComponent:
                if minHeight > height:
                    minHeight = height
                elif maxHeight < height:
                    maxHeight = height
                if minWidth > width:
                    minWidth = width
                elif maxWidth < width:
                    maxWidth = width
    return minHeight, minWidth,  maxHeight, maxWidth

# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate1.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here
    px_array = convertIMGtoGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # Create processed image
    px_array_processed = scaleTo0And255AndQuantize(px_array, image_width, image_height)
    px_array_processed = computeStandardDeviationImage5x5(px_array_processed, image_width, image_height)
    px_array_processed = scaleTo0And255AndQuantize(px_array_processed, image_width, image_height)
    px_array_processed = imageThreshholding(px_array_processed, image_width, image_height)

    for i in range(4):
        px_array_processed = computeDilation3x3(px_array_processed, image_width, image_height)
    for i in range(4):
        px_array_processed = computeErosion3x3(px_array_processed, image_width, image_height)

    # Find connected components and return the largest one
    (connectedCompArray, connectedCompDict) = computeConnectedComponentLabeling(px_array_processed, image_width, image_height)
    largestComponent = max(connectedCompDict, key=connectedCompDict.get)
    (bbox_min_y, bbox_min_x,  bbox_max_y, bbox_max_x) = computeBoundingBoxMinMax(connectedCompArray, largestComponent, image_width, image_height)

    # If the boundaries do not met the aspect ratio try the next largest component
    i = 2
    while (bbox_max_x - bbox_min_x)/(bbox_max_y - bbox_min_y) < 1.5 or (bbox_max_x - bbox_min_x)/(bbox_max_y - bbox_min_y) > 5:
        largestComponent = sorted(connectedCompDict, key=connectedCompDict.get)[-i]
        (bbox_min_y, bbox_min_x, bbox_max_y, bbox_max_x) = computeBoundingBoxMinMax(connectedCompArray, largestComponent, image_width,image_height)
        i += 1

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()