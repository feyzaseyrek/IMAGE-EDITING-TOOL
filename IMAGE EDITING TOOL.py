# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:45:21 2021

@author: feyzaseyrek
"""
from idlelib import window
from tkinter import *
import os
import PIL.Image
import ctypes
from PIL import ImageTk
from PIL import ImageOps
from tkinter import filedialog, Text
from tkinter.filedialog import askopenfilename
from PIL.Image import core as Image
from PIL import ImageDraw
from PIL import ImageChops
import imghdr
from collections import *
from PIL import Image, ImageFont
import tkinter.messagebox as tkMessageBox
from PIL import Image, ImageEnhance

############ INITIALIZE ##############
from skimage.viewer.utils import canvas
from tables.undoredo import redo, undo
from xlwings import apps
import cv2
import sys
import numpy as np


def init(root, window):
    buttonsInit(root, window)
    menuInit(root, window)
    window.data.image = None
    window.data.slope = None
    window.data.rotateWindowClose = False
    window.data.brightnessWClose = False
    window.data.brightnessLevel = None
    window.data.cropPopToHappen = False
    window.data.finish = False

    window.data.undoQueue = deque([], 10)
    window.data.redoQueue = deque([], 10)
    window.pack()


def buttonsInit(root, window):
    toolKitFrame = Frame(root)
    cropButton = Button(toolKitFrame, text="Crop", bg="#EED5D2", width=13, height=2, command=lambda: crop(window))
    cropButton.grid(row=0, column=9)
    rotateButton = Button(toolKitFrame, text="Rotate", bg="#E6E6FA", width=13, height=2, command=lambda: rotate(window))
    rotateButton.grid(row=0, column=4)
    brightnessButton = Button(toolKitFrame, text="Brightness", bg="#9ACD32", width=13, height=2,
                              command=lambda: brightness(window))
    brightnessButton.grid(row=0, column=3)
    mirrorButton = Button(toolKitFrame, text="Mirror", bg="#CD9B9B", width=13, height=2, command=lambda: mirror(window))
    mirrorButton.grid(row=0, column=0)
    invertButton = Button(toolKitFrame, text="Invert", bg="#FF8C00", width=13, height=2, command=lambda: invert(window))
    invertButton.grid(row=0, column=2)
    autocontrastButton = Button(toolKitFrame, text="Auto Contrast", bg="#EE3A8C", width=13, height=2,
                                command=lambda: autocontrast(window))
    autocontrastButton.grid(row=0, column=5)
    autoequalizeButton = Button(toolKitFrame, text="Auto Equalize", bg="#912CEE", width=13, height=2,
                                command=lambda: autoequalize(window))
    autoequalizeButton.grid(row=0, column=6)
    flipButton = Button(toolKitFrame, text="Flip", bg="#FFDE66", width=13, height=2, command=lambda: flip(window))
    flipButton.grid(row=0, column=1)
    autobrightnessButton = Button(toolKitFrame, text="Auto Brightness", bg="#800080", width=13, height=2,
                                  command=lambda: autobrightness(window))
    autobrightnessButton.grid(row=0, column=7)
    autosharpnessButton = Button(toolKitFrame, text="Auto Sharpness", bg="#FF7F50", width=13, height=2,
                                 command=lambda: autosharpness(window))
    autosharpnessButton.grid(row=0, column=8)

    warningButton = Button(toolKitFrame, text="Warning Filter", bg="#FEC163", width=13, height=2,
                           command=lambda: warning(window))
    warningButton.grid(row=1, column=0)
    cannyButton = Button(toolKitFrame, text="Canny Filter", bg="#CE9FFC", width=13, height=2,
                         command=lambda: canny(window))
    cannyButton.grid(row=1, column=1)
    shaepeningButton = Button(toolKitFrame, text="Shaepening Filter", bg="#FEC163", width=13, height=2,
                              command=lambda: shaepening(window))
    shaepeningButton.grid(row=1, column=2)
    coolingButton = Button(toolKitFrame, text="Cooling Filter", bg="#6078EA", width=13, height=2,
                           command=lambda: cooling(window))
    coolingButton.grid(row=1, column=3)
    constractButton = Button(toolKitFrame, text="Constract Filter", bg="#A0FE65", width=13, height=2,
                             command=lambda: constract(window))
    constractButton.grid(row=1, column=4)
    saturationButton = Button(toolKitFrame, text="Saturation Filter", bg="#622774", width=13, height=2,
                              command=lambda: saturation(window))
    saturationButton.grid(row=1, column=5)
    ccaButton = Button(toolKitFrame, text="CloneColor Filter", bg="#7367F0", width=13, height=2,
                       command=lambda: cca(window))
    ccaButton.grid(row=1, column=9)
    dilationButton = Button(toolKitFrame, text="Dilation Filter", bg="#DE4313", width=13, height=2,
                            command=lambda: dilation(window))
    dilationButton.grid(row=1, column=7)
    cartoonButton = Button(toolKitFrame, text="Cartoon Filter", bg="#622774", width=13, height=2,
                           command=lambda: cartoon(window))
    cartoonButton.grid(row=1, column=8)
    pencil_sketchButton = Button(toolKitFrame, text="Pencil Sketch Filter", bg="#A0FE65", width=13, height=2,
                                 command=lambda: pencil_sketch(window))
    pencil_sketchButton.grid(row=1, column=6)
    xproButton = Button(toolKitFrame, text="XPro Filter", bg="#FEC163", width=13, height=2,
                        command=lambda: xpro(window))
    xproButton.grid(row=1, column=4)
    moonButton = Button(toolKitFrame, text="Moon Filter", bg="#C53364", width=13, height=2,
                        command=lambda: moon(window))
    moonButton.grid(row=2, column=0)
    kelvinButton = Button(toolKitFrame, text="Kelvin Filter", bg="#1BCEDF", width=13, height=2,
                          command=lambda: kelvin(window))
    kelvinButton.grid(row=2, column=1)
    clarendonButton = Button(toolKitFrame, text="Clarendon Filter", bg="#5B247A", width=13, height=2,
                             command=lambda: clarendon(window))
    clarendonButton.grid(row=2, column=2)

    pencil_detailButton = Button(toolKitFrame, text="Pencil Detail Filter", bg="#FF7676", width=13, height=2,
                                 command=lambda: pencil_detail(window))
    pencil_detailButton.grid(row=2, column=3)

    toolKitFrame.pack(side=BOTTOM)


def menuInit(root, window):
    menubar = Menu(root)
    menubar.add_command(label="Open", command=lambda: newImage(window))
    menubar.add_command(label="Save", command=lambda: save(window))
    menubar.add_command(label="Reset", command=lambda: reset(window))

    editmenu = Menu(menubar, tearoff=0)
    editmenu.add_command(label="Undo   Z", command=lambda: undo(canvas))
    editmenu.add_command(label="Redo   Y", command=lambda: redo(canvas))
    root.config(menu=menubar)


############# MENU COMMANDS ########################################################################•


apps = []


def addApp():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File",  # select file from pc
                                          filetypes=(("executable", "*.exe"), ("all files", "*.*")))
    apps.append(filename)
    print(filename)
    for app in apps:  # run the select file from os
        os.startfile(app)


def save(window):
    if window.data.image != None:
        im = window.data.image
        im.save(window.data.imageLocation)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def newImage(window):
    imageName = askopenfilename()
    filetype = ""
    filetype = imghdr.what(imageName)
    #  except:
    #     tkMessageBox.showinfo(title="Image File",\
    #    message="Choose an Image File!" , parent=window.data.mainWindow)
    if filetype in ['jpeg', 'bmp', 'png', 'tiff']:
        window.data.imageLocation = imageName
        im = Image.open(imageName)
        window.data.image = im
        window.data.originalImage = im.copy()
        window.data.undoQueue.append(im.copy())
        window.data.imageSize = im.size  # Original Image dimensions
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)


def run():
    root = Tk()
    root.title("Image Editor program")
    windowWidth = 500
    windowHeight = 500
    window = Canvas(root, width=windowWidth, height=windowHeight, background="white")
    for app in apps:  # run the select file from os
        os.startfile(app)

    class Struct: pass

    window.data = Struct()
    window.data.width = 500
    window.data.height = 500
    window.data.mainWindow = root
    init(root, window)
    root.bind("<Key>", lambda event: keyPressed(window, event))
    root.mainloop()


#############################################################################

def reset(window):
    if window.data.image != None:
        window.data.image = window.data.originalImage.copy()
        save(window)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def mirror(window):
    if window.data.image != None:
        window.data.image = ImageOps.mirror(window.data.image)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)

    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def warning(window):
    if window.data.image != None:
        # Read image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # create a copy of the image to work on
        result = np.copy(image)

        # Original x-axis values
        originalValues = np.array([0, 50, 100, 150, 200, 255])
        # changes Y-axis values for red and blue channel
        redValues = np.array([0, 80, 150, 190, 220, 255])
        blueValues = np.array([0, 20, 40, 75, 150, 255])

        # create lookup table for red channel
        allValues = np.arange(0, 256)
        redLookupTable = np.interp(allValues, originalValues, redValues)

        # create lookup table for blue channel
        blueLookupTable = np.interp(allValues, originalValues, blueValues)

        # split into channels
        B, G, R = cv2.split(result)

        # apply mapping to red channel
        R = cv2.LUT(R, redLookupTable)
        R = np.uint8(R)

        # apply mapping to blue channel
        B = cv2.LUT(B, blueLookupTable)
        B = np.uint8(B)

        # mege the channels
        result = cv2.merge([B, G, R])

        # create windows to display images
        #  cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("warning", cv2.WINDOW_NORMAL)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("warning", result)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()

    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def canny(window):
    if window.data.image != None:
        image1 = cv2.imread(window.data.imageLocation, cv2.IMREAD_GRAYSCALE)
        image = np.copy(window.data.image)
        # define canny params
        lowThreshold = 50
        highTHreshold = 130
        # you can choose aperture size as 3 or 5 or 7
        apertureSize = 3

        # Blur the image before edge detection
        image = cv2.GaussianBlur(image, (3, 3), 0, 0)

        # apply canny
        output = cv2.Canny(image, lowThreshold, highTHreshold, apertureSize=apertureSize)

        # create windows to display images
        #  cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("canny", cv2.WINDOW_AUTOSIZE)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("canny", output)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()

    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def shaepening(window):
    if window.data.image != None:
        # read input image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # define sharpening kernel
        sharpeningKernel = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")

        # filter2D is used to perform the convolution.
        # The third parameter (depth) is set to -1 which means the bit-depth of the output image is the
        # same as the input image. So if the input image is of type CV_8UC3, the output image will also be of the same type
        output = cv2.filter2D(image, -1, sharpeningKernel)

        # create windows to display images
        #  cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("shaepening", cv2.WINDOW_AUTOSIZE)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("shaepening", output)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def cooling(window):
    if window.data.image != None:

        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # create a copy of the image to work on
        result = np.copy(window.data.image)

        # Original x-axis values
        originalValues = np.array([0, 50, 100, 150, 200, 255])
        # changes Y-axis values for red and blue channel
        redValues = np.array([0, 20, 40, 75, 150, 255])
        blueValues = np.array([0, 80, 150, 190, 220, 255])

        # create lookup table for red channel
        allValues = np.arange(0, 256)
        redLookupTable = np.interp(allValues, originalValues, redValues)

        # create lookup table for blue channel
        blueLookupTable = np.interp(allValues, originalValues, blueValues)

        # split into channels
        B, G, R = cv2.split(result)

        # apply mapping to red channel
        R = cv2.LUT(R, redLookupTable)
        R = np.uint8(R)

        # apply mapping to blue channel
        B = cv2.LUT(B, blueLookupTable)
        B = np.uint8(B)

        # mege the channels
        result = cv2.merge([B, G, R])

        # create windows to display images
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("cooling", cv2.WINDOW_NORMAL)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("cooling", result)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def constract(window):
    if window.data.image != None:
        # read image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # convert to YCrCb color space
        imageYcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # split into channels
        Y, C, B = cv2.split(imageYcb)

        # histogram equalization
        Y = cv2.equalizeHist(Y)

        # merge the channels
        imageYcb = cv2.merge([Y, C, B])

        # convert back to BGR color space
        result = cv2.cvtColor(imageYcb, cv2.COLOR_YCrCb2BGR)

        # create windows to display image
        #   cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("constract", cv2.WINDOW_NORMAL)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("constract", result)

        # press Esc to exit the program
        cv2.waitKey(0)

        # close all the opended windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def saturation(window):
    if window.data.image != None:
        # read image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # convert to HSV color space
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # create a copy of hsv image to work on
        hsvImageCopy = hsvImage.copy()

        # convert to float32
        hsvImageCopy = np.float32(hsvImageCopy)

        # initialize desaturation Scale value
        saturationScale = 0.01

        # split the channels
        H, S, V = cv2.split(hsvImageCopy)

        # Desaturation. Multiply S channel by scaling factor and make sure the values are between 0 and 255
        S = np.clip(S * saturationScale, 0, 255)

        # merge the channels
        hsvImageCopy = cv2.merge([H, S, V])

        # convert back to uint8
        hsvImageCopy = np.uint8(hsvImageCopy)

        # convert back to bgr color space
        hsvImageCopy = cv2.cvtColor(hsvImageCopy, cv2.COLOR_HSV2BGR)

        # create windows to show image
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
        cv2.namedWindow("desaturated", cv2.WINDOW_NORMAL)

        # cv2.imshow("image", image)
        cv2.imshow("hsv", hsvImage)
        cv2.imshow("desaturated", hsvImageCopy)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def cca(window):
    if window.data.image != None:
        # Read image as grayScale over which cca is to be applied

        image1 = np.copy(window.data.image)
        image = cv2.imread(window.data.imageLocation, cv2.IMREAD_GRAYSCALE)

        # get binary image
        th, binaryImage = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Find connected components
        _, binaryImage = cv2.connectedComponents(binaryImage)

        # get clone of binary image to work on so that finally we can compare input and output images
        binaryImageClone = np.copy(binaryImage)

        # Find the max and min pixel values and their locations
        (minValue, maxValue, minPosition, maxPosition) = cv2.minMaxLoc(binaryImageClone)

        # normalize the image so that the min value is 0 and max value is 255
        binaryImageClone = 255 * (binaryImageClone - minValue) / (maxValue - minValue)

        # convert image to 8bits unsigned type
        binaryImageClone = np.uint8(binaryImageClone)

        # Apply a color map
        binaryImageCloneColorMap = cv2.applyColorMap(binaryImageClone, cv2.COLORMAP_JET)

        # Create windows to display images
        # cv2.namedWindow("input image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("cca image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("cca image color", cv2.WINDOW_NORMAL)

        # Display images
        # cv2.imshow("input image", image)
        cv2.imshow("cca image", binaryImageClone)
        cv2.imshow("cca image color", binaryImageCloneColorMap)

        # Press esc on keybaord to exit the program
        cv2.waitKey(0)

        # Close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def dilation(window):
    if window.data.image != None:
        # read the image to be dilated
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # check if the input image exits
        if window.data.image is None:
            print("Input image not found")
            sys.exit()

        # Create a structiong element/kernel which will be used for dilation
        dilationSize = 6
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * dilationSize + 1, 2 * dilationSize + 1),
                                            (dilationSize, dilationSize))

        # Apply dilate function on input image. Dilation will increase brightness, First Parameter is the original image,
        # second is the dilated image
        dilatedImage = cv2.dilate(image, element)

        # Create windows to diaplay images
        #  cv2.namedWindow("input image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dilated image", cv2.WINDOW_NORMAL)

        # Display the images
        # cv2.imshow("input image", image)
        cv2.imshow("dilated image", dilatedImage)

        # Press esc on keyboard to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def cartoon(window):
    if window.data.image != None:
        # read image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # check if image exists
        if window.data.image is None:
            print("can not find image")
            exit()

        # convert to gray scale
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur
        grayImage = cv2.GaussianBlur(grayImage, (3, 3), 0)

        # detect edges
        edgeImage = cv2.Laplacian(grayImage, -1, ksize=5)
        edgeImage = 255 - edgeImage

        # threshold image
        ret, edgeImage = cv2.threshold(edgeImage, 150, 255, cv2.THRESH_BINARY)

        # blur images heavily using edgePreservingFilter
        edgePreservingImage = cv2.edgePreservingFilter(image, flags=2, sigma_s=50, sigma_r=0.4)

        # create output matrix
        output = np.zeros(grayImage.shape)

        # combine cartoon image and edges image
        output = cv2.bitwise_and(edgePreservingImage, edgePreservingImage, mask=edgeImage)

        # create windows to display images
        #  cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("cartoon", cv2.WINDOW_AUTOSIZE)

        # display images
        #  cv2.imshow("image", image)
        cv2.imshow("cartoon", output)
        # return image,output

        # press esc to exit program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def pencil_sketch(window):
    if window.data.image != None:
        # read image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # convert to gray scale
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # invert the gray image
        grayImageInv = 255 - grayImage

        # Apply gaussian blur
        grayImageInv = cv2.GaussianBlur(grayImageInv, (21, 21), 0)

        # blend using color dodge
        output = cv2.divide(grayImage, 255 - grayImageInv, scale=256.0)

        # create windows to dsiplay images
        #  cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("pencilsketch", cv2.WINDOW_AUTOSIZE)

        # display images
        # cv2.imshow("image",  image)
        cv2.imshow("pencilsketch", output)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def xpro(window):
    if window.data.image != None:
        # Read input image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # create a copy of input image to work on
        output = image.copy()

        # split into channels
        B, G, R = cv2.split(output)

        # define vignette scale
        vignetteScale = 6

        # calculate the kernel size
        k = np.min([output.shape[1], output.shape[0]]) / vignetteScale

        # create kernel to get the Halo effect
        kernelX = cv2.getGaussianKernel(output.shape[1], k)
        kernelY = cv2.getGaussianKernel(output.shape[0], k)
        kernel = kernelY * kernelX.T

        # normalize the kernel
        mask = cv2.normalize(kernel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # apply halo effect to all the three channels of the image
        B = B + B * mask
        G = G + G * mask
        R = R + R * mask

        # merge back the channels
        output = cv2.merge([B, G, R])

        output = output / 2

        # limit the values between 0 and 255
        output = np.clip(output, 0, 255)

        # convert back to uint8
        output = np.uint8(output)

        # split the channels
        B, G, R = cv2.split(output)

        # Interpolation values
        redValuesOriginal = np.array([0, 42, 105, 148, 185, 255])
        redValues = np.array([0, 28, 100, 165, 215, 255])
        greenValuesOriginal = np.array([0, 40, 85, 125, 165, 212, 255])
        greenValues = np.array([0, 25, 75, 135, 185, 230, 255])
        blueValuesOriginal = np.array([0, 40, 82, 125, 170, 225, 255])
        blueValues = np.array([0, 38, 90, 125, 160, 210, 222])

        # create lookuptable
        allValues = np.arange(0, 256)

        # create lookup table for red channel
        redLookuptable = np.interp(allValues, redValuesOriginal, redValues)
        # apply the mapping for red channel
        R = cv2.LUT(R, redLookuptable)

        # create lookup table for green channel
        greenLookuptable = np.interp(allValues, greenValuesOriginal, greenValues)
        # apply the mapping for red channel
        G = cv2.LUT(G, greenLookuptable)

        # create lookup table for blue channel
        blueLookuptable = np.interp(allValues, blueValuesOriginal, blueValues)
        # apply the mapping for red channel
        B = cv2.LUT(B, blueLookuptable)

        # merge back the channels
        output = cv2.merge([B, G, R])

        # convert back to uint8
        output = np.uint8(output)

        # adjust contrast
        # convert to YCrCb color space
        output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)

        # convert to float32
        output = np.float32(output)

        # split the channels
        Y, Cr, Cb = cv2.split(output)

        # scale the Y channel
        Y = Y * 1.2

        # limit the values between 0 and 255
        Y = np.clip(Y, 0, 255)

        # merge back the channels
        output = cv2.merge([Y, Cr, Cb])

        # convert back to uint8
        output = np.uint8(output)

        # convert back to BGR color space
        output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)

        # create window to display images
        #  cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("xpro", cv2.WINDOW_AUTOSIZE)

        # display images
        #  cv2.imshow("image", image)
        cv2.imshow("xpro", output)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def moon(window):
    if window.data.image != None:
        # Read input image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # create a clone of input image to work on
        output = image.copy()

        # convert to LAB color space
        output = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)

        # split into channels
        L, A, B = cv2.split(output)

        # Interpolation values
        originalValues = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255])
        values = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255])

        # create lookup table
        allValues = np.arange(0, 256)

        # Creating the lookuptable
        lookuptable = np.interp(allValues, originalValues, values)

        # apply mapping for L channels
        L = cv2.LUT(L, lookuptable)

        # convert to uint8
        L = np.uint8(L)

        # merge back the channels
        output = cv2.merge([L, A, B])

        # convert back to BGR color space
        output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)

        # desaturate the image
        # convert to HSV color space
        output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

        # split into channels
        H, S, V = cv2.split(output)

        # Multiply S channel by saturation scale value
        S = S * 0.01

        # convert to uint8
        S = np.uint8(S)

        # limit the values between 0 and 256
        S = np.clip(S, 0, 255)

        # merge back the channels
        output = cv2.merge([H, S, V])

        # convert back to BGR color space
        output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

        # create windows to display images
        #  cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("moon", cv2.WINDOW_AUTOSIZE)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("moon", output)

        # press esc to exit the program
        cv2.waitKey(0)

        # destroy all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def kelvin(window):
    if window.data.image != None:
        # read input image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # create a copy of input image to work on
        output = image.copy()

        # split the channels
        blueChannel, greenChannel, redChannel = cv2.split(output)

        # Interpolation values
        redValuesOriginal = np.array([0, 60, 110, 150, 235, 255])
        redValues = np.array([0, 102, 185, 220, 245, 245])
        greenValuesOriginal = np.array([0, 68, 105, 190, 255])
        greenValues = np.array([0, 68, 120, 220, 255])
        blueValuesOriginal = np.array([0, 88, 145, 185, 255])
        blueValues = np.array([0, 12, 140, 212, 255])

        # create lookup table
        allValues = np.arange(0, 256)
        # Creating the lookuptable for blue channel
        blueLookuptable = np.interp(allValues, blueValuesOriginal, blueValues)
        # Creating the lookuptable for green channel
        greenLookuptable = np.interp(allValues, greenValuesOriginal, greenValues)
        # Creating the lookuptable for red channel
        redLookuptable = np.interp(allValues, redValuesOriginal, redValues)

        # Apply the mapping for blue channel
        blueChannel = cv2.LUT(blueChannel, blueLookuptable)
        # Apply the mapping for green channel
        greenChannel = cv2.LUT(greenChannel, greenLookuptable)
        # Apply the mapping for red channel
        redChannel = cv2.LUT(redChannel, redLookuptable)

        # merging back the channels
        output = cv2.merge([blueChannel, greenChannel, redChannel])

        # convert to uint8
        output = np.uint8(output)

        # create windows to display images
        #  cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Kelvin", cv2.WINDOW_AUTOSIZE)

        # display images
        #  cv2.imshow("image", image)
        cv2.imshow("Kelvin", output)

        # press esc to exit the program
        cv2.waitKey(0)

        # destroy all windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def clarendon(window):
    if window.data.image != None:

        # read input image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        if image is None:
            print("can not find image")
            sys.exit()

        # create a copy of input image to work on
        clarendon = image.copy()

        # split the channels
        blueChannel, greenChannel, redChannel = cv2.split(clarendon)

        # Interpolation values
        originalValues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
        blueValues = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255])
        redValues = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249])
        greenValues = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255])

        # Creating the lookuptables
        fullRange = np.arange(0, 256)
        # Creating the lookuptable for blue channel
        blueLookupTable = np.interp(fullRange, originalValues, blueValues)
        # Creating the lookuptables for green channel
        greenLookupTable = np.interp(fullRange, originalValues, greenValues)
        # Creating the lookuptables for red channel
        redLookupTable = np.interp(fullRange, originalValues, redValues)

        # Apply the mapping for blue channel
        blueChannel = cv2.LUT(blueChannel, blueLookupTable)
        # Apply the mapping for green channel
        greenChannel = cv2.LUT(greenChannel, greenLookupTable)
        # Apply the mapping for red channel
        redChannel = cv2.LUT(redChannel, redLookupTable)

        # merging back the channels
        clarendon = cv2.merge([blueChannel, greenChannel, redChannel])

        # convert to uint8
        clarendon = np.uint8(clarendon)

        # create windows to display images
        # cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("clarendon", cv2.WINDOW_AUTOSIZE)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("clarendon", clarendon)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def pencil_detail(window):
    if window.data.image != None:
        # read image
        image1 = cv2.imread(window.data.imageLocation)
        image = np.copy(window.data.image)

        # check if images exists
        if image is None:
            print("can not find image")
            sys.exit()

        # convert to gray scale
        output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur
        output = cv2.GaussianBlur(output, (3, 3), 0)

        # detect edges in the image
        output = cv2.Laplacian(output, -1, ksize=5)

        # invert the binary image
        output = 255 - output

        # binary thresholding
        ret, output = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY)

        # create widnows to dispplay images
        # cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("pencilsketch", cv2.WINDOW_AUTOSIZE)

        # display images
        # cv2.imshow("image", image)
        cv2.imshow("pencilsketch", output)

        # press esc to exit the program
        cv2.waitKey(0)

        # close all the opened windows
        cv2.destroyAllWindows()
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def flip(window):
    if window.data.image != None:
        window.data.image = ImageOps.flip(window.data.image)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def autocontrast(window):
    if window.data.image != None:
        window.data.image = ImageOps.autocontrast(window.data.image, cutoff=5, ignore=5)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def crop(window):
    window.data.cropPopToHappen = True
    if window.data.image != None:
        tkMessageBox.showinfo(title="WARNING", message="Please, Draw  area to crop  and then  press Enter",
                              parent=window.data.mainWindow)
        window.data.mainWindow.bind("<ButtonPress-1>", lambda event: begining(event, window))
        window.data.mainWindow.bind("<ButtonRelease-1>", lambda event: finish(event, window))
        window.data.finish = False
    elif window.data.image == None:
        tkMessageBox.showinfo(title="Warning", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def begining(event, window):
    if window.data.finish == False and window.data.cropPopToHappen == True:
        window.data.beginingX = event.x
        window.data.beginingY = event.y


def finish(event, window):
    if window.data.cropPopToHappen == True:
        window.data.finishX = event.x
        window.data.finishY = event.y
        window.create_rectangle(window.data.beginingX, window.data.beginingY, window.data.finishX, window.data.finishY,
                                outline="black", width=2)
        window.data.mainWindow.bind("<Return>", lambda event: Cropping(event, window))
        window.data.finish = False


def Cropping(event, window):
    window.data.image = window.data.image.crop(
        (int(round((window.data.beginingX - window.data.imageTopX) * window.data.imageScale)),
         int(round((window.data.beginingY - window.data.imageTopY) * window.data.imageScale)),
         int(round((window.data.finishX - window.data.imageTopX) * window.data.imageScale)),
         int(round((window.data.finishY - window.data.imageTopY) * window.data.imageScale))))
    window.data.cropPopToHappen = False
    window.data.finish = False
    # save(window)
    window.data.undoQueue.append(window.data.image.copy())
    window.data.imageForTk = makeImageForTk(window)
    createImage(window)


def rotatecomplete(window, rwindow, rotateSlider, firstAngle):
    if window.data.image != None and rwindow.winfo_exists():
        window.data.slope = rotateSlider.get()
        if window.data.slope != None and window.data.slope != firstAngle:
            window.data.image = window.data.image.rotate(float(window.data.slope))
            window.data.imageForTk = makeImageForTk(window)
            createImage(window)
    window.after(200, lambda: rotatecomplete(window, rwindow, rotateSlider, window.data.slope))


# def exitWindow(window):
#       window.data.undoQueue.append(window.data.image.copy())
#      window.data.rotateWindowClose=True

def rotate(window):
    if window.data.image != None:
        rwindow = Toplevel(window.data.mainWindow)
        rwindow.title("Rotation adjustment")
        rotateSlider = Scale(rwindow, from_=0, to=360, orient=VERTICAL)
        rotateSlider.pack()
        tamamRotateFrame = Frame(rwindow)
        # tamamRotateButton=Button(tamamRotateFrame, text="Apply",command=lambda: exitWindow(window))
        # tamamRotateButton.grid(row=0,column=0)
        # tamamRotateFrame.pack(side=TOP)
        rotatecomplete(window, rwindow, rotateSlider, 0)

    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def changeBrightness(window, bwindow, brightnessSlider, firstVal):
    if window.data.brightnessWClose == True:
        bwindow.destroy()
        window.data.brightnessWClose = False
    else:

        if window.data.image != None and bwindow.winfo_exists():
            sliderVal = brightnessSlider.get()
            scale = (sliderVal - firstVal) / 100.0
            window.data.image = window.data.image.point(lambda i: i + int(round(i * scale)))
            window.data.imageForTk = makeImageForTk(window)
            createImage(window)
            window.after(200, lambda: changeBrightness(window, bwindow, brightnessSlider, sliderVal))


def brightness(window):
    if window.data.image != None:

        bwindow = Toplevel(window.data.mainWindow)
        bwindow.title("Brightness adjustment")
        brightnessSlider = Scale(bwindow, from_=-100, to=100, orient=VERTICAL)
        brightnessSlider.pack()
        tamamBrightnessFrame = Frame(bwindow)
        tamamBrightnessButton = Button(tamamBrightnessFrame, text="apply", command=lambda: tamamtheWindow(window))
        tamamBrightnessButton.grid(row=0, column=0)
        tamamBrightnessFrame.pack(side=TOP)
        changeBrightness(window, bwindow, brightnessSlider, 0)
        brightnessSlider.set(0)

    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def tamamtheWindow(window):
    window.data.undoQueue.append(window.data.image.copy())
    window.data.brightnessWClose = True


def autoequalize(window):
    if window.data.image != None:
        window.data.image = ImageOps.equalize(window.data.image)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def autobrightness(window):
    if window.data.image != None:
        window.data.image = ImageEnhance.Brightness(window.data.image).enhance(1.5)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def autosharpness(window):
    if window.data.image != None:
        window.data.image = ImageEnhance.Sharpness(window.data.image).enhance(2)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def invert(window):
    if window.data.image != None:
        window.data.image = ImageOps.invert(window.data.image)
        window.data.imageForTk = makeImageForTk(window)
        createImage(window)
    elif window.data.image == None:
        tkMessageBox.showinfo(title="WARNING", message="Click  the 'Open' to upload an image, please!",
                              parent=window.data.mainWindow)


def makeImageForTk(window):
    image = window.data.image
    Width = window.data.image.size[0]
    Height = window.data.image.size[1]
    if Width > Height:
        newsize = image.resize((window.data.width, int(round(float(Height) * window.data.width / Width))))
        window.data.imageScale = float(Width) / window.data.width
    else:
        newsize = image.resize((int(round(float(Width) * window.data.height / Height)), window.data.height))
        window.data.imageScale = float(Height) / window.data.height
        window.data.newsizeimage = newsize
    window.data.newsizeimage = newsize
    return ImageTk.PhotoImage(newsize)


def createImage(window):
    window.create_image(window.data.width / 2.5 - window.data.newsizeimage.size[0] / 2.5,
                        window.data.height / 2.5 - window.data.newsizeimage.size[1] / 2.5,
                        anchor=NW, image=window.data.imageForTk)
    window.data.imageTopX = int(round(window.data.width / 2.5 - window.data.newsizeimage.size[0] / 2.5))
    window.data.imageTopY = int(round(window.data.height / 2.5 - window.data.newsizeimage.size[1] / 2.5))


run()




