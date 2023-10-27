import cv2
import tkinter as tk
from tkinter import filedialog, Entry, Label, Frame, Button, colorchooser
from PIL import Image, ImageTk
import numpy as np

class ImageProcessingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")
        self.original_image = None
        self.modified_image = None
        self.color_lower_bound = None
        self.color_upper_bound = None

        self.load_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.load_button.pack()

        self.process_button = tk.Button(root, text="Process Image", command=self.process_image)
        self.process_button.pack()
        self.process_button["state"] = "disabled"

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        self.original_label = tk.Label(self.image_frame, text="Original Image")
        self.original_label.grid(row=0, column=0)
        self.modified_label = tk.Label(self.image_frame, text="Modified Image")
        self.modified_label.grid(row=0, column=1)

        self.preview_label_original = tk.Label(self.image_frame, width=500, height=500)
        self.preview_label_original.grid(row=1, column=0)
        self.preview_label_modified = tk.Label(self.image_frame, width=500, height=500)
        self.preview_label_modified.grid(row=1, column=1)

        self.operations_frame = tk.Frame(root)
        self.operations_frame.pack()

        self.transformations_frame = tk.Frame(root)
        self.transformations_frame.pack()

        self.rotation_label = tk.Label(self.transformations_frame, text="Rotation Angle:")
        self.rotation_angle_entry = Entry(self.transformations_frame)
        self.rotation_button = tk.Button(self.transformations_frame, text="Rotation", command=self.rotate)
        self.crop_button = tk.Button(self.transformations_frame, text="Crop", command=self.crop)
        self.flip_button = tk.Button(self.transformations_frame, text="Flip", command=self.flip)

        self.rotation_label.grid(row=0, column=0)
        self.rotation_angle_entry.grid(row=0, column=1)
        self.rotation_button.grid(row=0, column=2)
        self.crop_button.grid(row=0, column=3)
        self.flip_button.grid(row=0, column=4)

        self.color_button = tk.Button(self.operations_frame, text="Color", command=self.color)
        self.bw_button = tk.Button(self.operations_frame, text="Black & White", command=self.bw)
        self.grayscale_button = tk.Button(self.operations_frame, text="Grayscale", command=self.grayscale)

        self.color_button.grid(row=0, column=0)
        self.bw_button.grid(row=0, column=1)
        self.grayscale_button.grid(row=0, column=2)

        self.filters_frame = tk.Frame(root)
        self.filters_frame.pack()

        self.filters_label = Label(self.filters_frame, text="Filters")
        self.filters_label.pack()

        self.sharpen_button = tk.Button(self.filters_frame, text="Sharpen", command=self.sharpen)
        self.smooth_button = tk.Button(self.filters_frame, text="Smooth", command=self.smooth)
        self.edge_detection_button = tk.Button(self.filters_frame, text="Edge Detection", command=self.edge_detection)

        self.sharpen_button.pack(side="left")
        self.smooth_button.pack(side="left")
        self.edge_detection_button.pack(side="left")

        self.segmentation_frame = tk.Frame(root)
        self.segmentation_frame.pack()
        self.region_based_button = tk.Button(self.segmentation_frame, text="Region-Based Segmentation", command=self.region_based_segmentation)
        self.region_based_button.pack()

        self.adjustments_frame = tk.Frame(root)
        self.adjustments_frame.pack()

        self.tonal_button = tk.Button(self.adjustments_frame, text="Tonal Transformations", command=self.tonal_transform)
        self.color_balance_button = tk.Button(self.adjustments_frame, text="Color Balancing", command=self.color_balance)

        self.tonal_button.pack()
        self.color_balance_button.pack()

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
        self.modified_image = cv2.warpAffine(self.original_image, rotation_matrix, (self.original_image.shape[1], self.original_image.shape[0]))
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

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessingTool(root)
    root.mainloop()


