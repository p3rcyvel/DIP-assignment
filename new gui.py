import cv2
import tkinter as tk
from tkinter import filedialog, Entry, Label, Frame, Button, colorchooser
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the SRCNN model
def create_srcnn_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 1)))
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(layers.Conv2D(1, (5, 5), activation='linear', padding='same'))
    return model

class ImageProcessingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")

        self.create_menu()
        self.create_gui()

        self.srcnn_model = create_srcnn_model()
        self.original_image = None
        self.modified_image = None

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

    def create_gui(self):
        self.create_buttons()
        self.create_images_display()
        self.create_operations_frame()

    def create_buttons(self):
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)

        self.load_button = Button(button_frame, text="Open Image", command=self.open_image)
        self.load_button.grid(row=0, column=0, padx=5)
        self.process_button = Button(button_frame, text="Process Image", command=self.process_image)
        self.process_button.grid(row=0, column=1, padx=5)
        self.process_button["state"] = "disabled"

    def create_images_display(self):
        image_frame = Frame(self.root)
        image_frame.pack()

        self.original_label = Label(image_frame, text="Original Image")
        self.original_label.grid(row=0, column=0)
        self.modified_label = Label(image_frame, text="Modified Image")
        self.modified_label.grid(row=0, column=1)

        self.preview_label_original = Label(image_frame, width=500, height=500)
        self.preview_label_original.grid(row=1, column=0)
        self.preview_label_modified = Label(image_frame, width=500, height=500)
        self.preview_label_modified.grid(row=1, column=1)

    def create_operations_frame(self):
        operations_frame = Frame(self.root)
        operations_frame.pack()

        self.create_transformations_widgets(operations_frame)
        self.create_filters_widgets(operations_frame)
        self.create_segmentation_widgets(operations_frame)
        self.create_adjustments_widgets(operations_frame)

    def create_transformations_widgets(self, parent_frame):
        transformations_frame = Frame(parent_frame)
        transformations_frame.pack(pady=10)

        self.rotation_label = Label(transformations_frame, text="Rotation Angle:")
        self.rotation_label.grid(row=0, column=0)
        self.rotation_angle_entry = Entry(transformations_frame)
        self.rotation_angle_entry.grid(row=0, column=1)
        self.rotation_button = Button(transformations_frame, text="Rotate", command=self.rotate)
        self.rotation_button.grid(row=0, column=2)
        self.crop_button = Button(transformations_frame, text="Crop", command=self.crop)
        self.crop_button.grid(row=0, column=3)
        self.flip_button = Button(transformations_frame, text="Flip", command=self.flip)
        self.flip_button.grid(row=0, column=4)

    def create_filters_widgets(self, parent_frame):
        filters_frame = Frame(parent_frame)
        filters_frame.pack()

        filters_label = Label(filters_frame, text="Filters")
        filters_label.grid(row=0, column=0)

        self.sharpen_button = Button(filters_frame, text="Sharpen", command=self.sharpen)
        self.sharpen_button.grid(row=1, column=0)
        self.smooth_button = Button(filters_frame, text="Smooth", command=self.smooth)
        self.smooth_button.grid(row=1, column=1)
        self.edge_detection_button = Button(filters_frame, text="Edge Detection", command=self.edge_detection)
        self.edge_detection_button.grid(row=1, column=2)
        self.emboss_button = Button(filters_frame, text="Emboss", command=self.emboss)
        self.emboss_button.grid(row=1, column=3)

    def create_segmentation_widgets(self, parent_frame):
        segmentation_frame = Frame(parent_frame)
        segmentation_frame.pack(pady=10)

        self.region_based_button = Button(segmentation_frame, text="Region-Based Segmentation", command=self.region_based_segmentation)
        self.region_based_button.grid(row=0, column=0)

    def create_adjustments_widgets(self, parent_frame):
        adjustments_frame = Frame(parent_frame)
        adjustments_frame.pack()

        self.tonal_button = Button(adjustments_frame, text="Tonal Transformations", command=self.tonal_transform)
        self.tonal_button.grid(row=0, column=0)
        self.color_balance_button = Button(adjustments_frame, text="Color Balancing", command=self.color_balance)
        self.color_balance_button.grid(row=0, column=1)
        self.image_enhance_button = Button(adjustments_frame, text="Image Enhancement", command=self.image_enhancement)
        self.image_enhance_button.grid(row=0, column=2)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.modified_image = self.original_image.copy()
            self.show_image(self.original_image, self.preview_label_original)
            self.show_image(self.modified_image, self.preview_label_modified)
            self.process_button["state"] = "active"

    def show_image(self, image, label):
        if image is not None:
            # Resize the image to fit the 500x500 window
            scale_factor = 500 / max(image.shape[0], image.shape[1])
            image_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            label.config(image=image_tk)
            label.image = image_tk

    def process_image(self):
        if self.modified_image is not None:
            self.show_image(self.modified_image, self.preview_label_modified)

    # Existing functions (rotate, crop, flip, etc.)
    def color(self):
        self.modified_image = self.original_image.copy()
        self.show_image(self.modified_image, self.preview_label_modified)

    def bw(self):
        self.modified_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.show_image(self.modified_image, self.preview_label_modified)

    def grayscale(self):
        self.modified_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.show_image(self.modified_image, self.preview_label_modified)

    def sharpen(self):
        kernel = np.array([-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1])
        self.modified_image = cv2.filter2D(self.original_image, -1, kernel)
        self.show_image(self.modified_image, self.preview_label_modified)

    def smooth(self):
        self.modified_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        self.show_image(self.modified_image, self.preview_label_modified)

    def edge_detection(self):
        if self.modified_image is not None:
            # Implement edge detection using the Canny algorithm
            self.modified_image = cv2.Canny(self.original_image, 100, 200)
            self.show_image(self.modified_image, self.preview_label_modified)
            
    def emboss(self):
        if self.modified_image is not None:
        # Define the embossing kernel
            kernel = np.array([[-2, -1, 0],
                             [-1, 1, 1],
                             [0, 1, 2]])

        # Apply the embossing filter
            self.modified_image = cv2.filter2D(self.modified_image, -1, kernel)
            self.show_image(self.modified_image, self.preview_label_modified)

    def region_based_segmentation(self):
        if self.original_image is not None:
            color = colorchooser.askcolor()[0]  # Open a color picker dialog and get the chosen color
            if color is not None:
                lower_bound = np.array([int(c) for c in color])
                upper_bound = np.array([int(c) for c in color])
                mask = cv2.inRange(self.original_image, lower_bound, upper_bound)
                self.modified_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
                self.show_image(self.modified_image, self.preview_label_modified)

    def rotate(self):
        angle = float(self.rotation_angle_entry.get())
        rotation_matrix = cv2.getRotationMatrix2D((self.original_image.shape[1] / 2, self.original_image.shape[0] / 2), angle, 1)
        self.modified_image = cv2.warpAffine(self.original_image, rotation_matrix, self.original_image.shape[1], self.original_image.shape[0])
        self.show_image(self.modified_image, self.preview_label_modified)

    def crop(self):
        left = 50  # You can change these values as needed
        top = 50
        right = 200
        bottom = 200
        self.modified_image = self.original_image[top:bottom, left:right]
        self.show_image(self.modified_image, self.preview_label_modified)

    def flip(self):
        self.modified_image = cv2.bitwise_not(self.original_image)
        self.show_image(self.modified_image, self.preview_label_modified)

    def tonal_transform(self):
        if self.modified_image is not None:
            # Implement tonal transformations on self.modified_image here
            # Example: Adjust contrast and brightness
            alpha = 2.0  # Adjust contrast (1.0 is the original contrast)
            beta = 50   # Adjust brightness (0 is the original brightness)
            self.modified_image = cv2.convertScaleAbs(self.modified_image, alpha=alpha, beta=beta)
            self.show_image(self.modified_image, self.preview_label_modified)

    def color_balance(self):
        if self.modified_image is not None:
            # Implement color balancing on self.modified_image here
            # Example: Equalize the histogram for each color channel
            blue, green, red = cv2.split(self.modified_image)
            blue = cv2.equalizeHist(blue)
            green = cv2.equalizeHist(green)
            red = cv2.equalizeHist(red)
            self.modified_image = cv2.merge((blue, green, red))
            self.show_image(self.modified_image, self.preview_label_modified)

    def image_enhancement(self):
        if self.original_image is not None:
            # Implement image enhancement using the SRCNN model
           # gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            #enhanced_image = self.srcnn_model.predict(gray_image.reshape(1, gray_image.shape[0], gray_image.shape[1], 1))
           # enhanced_image = enhanced_image[0].astype(np.uint8)
            #self.modified_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            #self.show_image(self.modified_image, self.preview_label_modified)

   
        # Convert the original image to grayscale
         gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image to the original size
         resized_gray_image = cv2.resize(gray_image, (self.original_image.shape[1], self.original_image.shape[0]))

        # Predict the enhanced grayscale image using the SRCNN model
         enhanced_gray_image = self.srcnn_model.predict(resized_gray_image.reshape(1, resized_gray_image.shape[0], resized_gray_image.shape[1], 1))
         enhanced_gray_image = enhanced_gray_image[0].astype(np.uint8)

        # Convert the enhanced grayscale image back to color
         self.modified_image = cv2.cvtColor(enhanced_gray_image, cv2.COLOR_GRAY2BGR)

        # Display the enhanced image
        self.show_image(self.modified_image, self.preview_label_modified)


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessingTool(root)
    root.mainloop()
