
import tkinter
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from models import create_model
from options.test_options import TestOptions
import pickle
from util import util
from scipy.misc import imresize

top = Tk()
top.title = 'Image Dehazing'
top.geometry('550x500') # window size
canvas = Canvas(top, width=300, height=300, bd=0, bg='white') # the loading and display area for the image
canvas.grid(row=1, column=0)

def save_images(visuals, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)

        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')

    return im

def showImg():
    File = filedialog.askopenfilename(title='Open Image')
    e.set(File)
    load = Image.open(e.get())
    load = load.resize((256, 256))
    imgfile = ImageTk.PhotoImage(load)
    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(2, 2, anchor='nw', image=imgfile)


e = StringVar()

submit_button = Button(top, text='Select the hazy Image', command=showImg) # calling the show image button
submit_button.grid(row=0, column=0)

def Predict():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create s
    pickle_in = open("emb.pickle", "rb")
    emb = pickle.load(pickle_in)
    for data in emb:
        if data['B_paths'][0].split('/')[-1] != e.get().split('/')[-1]: continue # compare with the user selected file to get the embedding

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        im = save_images(visuals)

        img = Image.fromarray(im)
        img = ImageTk.PhotoImage(img)
        canvas.image = img  # <--- keep reference of your image
        canvas.create_image(2, 2, anchor='nw', image=img)

submit_button = Button(top, text='Attempt to dehaze', command=Predict) # predicting the image
submit_button.grid(row=2, column=0)

l1 = Label(top, text='Please <Open> a RGB image, then press <Predict> to see the dehazed results.')
l1.grid(row=3)

t1 = Text(top, bd=0, width=20, height=10, font='Fixdsys -14')
t1.grid(row=4, column=4)
top.mainloop()