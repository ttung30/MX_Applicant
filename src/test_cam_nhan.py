import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import tkinter as tk
import os
from tkinter import filedialog
from src.alex_net import Alex_net
import tensorflow as tf
import pandas as pd

model = Alex_net(11)
dummy_x = tf.zeros((1, 224, 224, 3))
model._set_inputs(dummy_x)
loaded_model = tf.keras.models.load_model('model')
train_df = pd.read_csv('/home/tung/Downloads/mx/fc-cnn-assignment/train.csv')


window = tk.Tk()
window.geometry("1280x720")
window.title("MX")
window.configure(bg="#FFFFFF")  
pred_labels = 0
file_path = ''
def cv2_to_pil(cv_img):
    return PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
# Function to open folder
def open_folder():
    global file_path
    file_path = filedialog.askopenfilename()
    print("Selected file:", file_path)
    # Add your logic to handle the selected folder here
# Add logo image to top-left corner
logo_path = ""  # Replace with the path to your logo image
if os.path.exists(logo_path):
    logo_image = PIL.Image.open(logo_path)
    logo_image = logo_image.resize((100, 100), PIL.Image.LANCZOS)
    logo_photo = PIL.ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(window, image=logo_photo, bg="#FFFFFF")  
    logo_label.place(x=10, y=10)
else:
    print("Logo image not found at specified path.")

# Label for Project Title
project_title = tk.Label(window, text="MX's Candidate: Nguyen Thanh Tung", bg="white", font=("Times New Roman", 26))
project_title.place(relx=0.5, rely=0.008, anchor=tk.N)
# Frame for video display
frame = tk.Frame(window, bg="#D9D9D9")
frame.place(relx=0.65, rely=0.5, relwidth=0.8, relheight=0.7, anchor=tk.CENTER)

canvas = tk.Canvas(frame, bg="#FFFFFF", bd=0, highlightthickness=0)
canvas.place(relx=0.5, rely=0.5, relwidth=1, relheight=1, anchor=tk.CENTER)

# Placeholder for pred_labels display
pred_label_display = tk.Label(window, text="", font=("Times New Roman", 20), bg="white")
pred_label_display.place(x=1000, y=400)
real_label_display = tk.Label(window, text="", font=("Times New Roman", 20), bg="white")
real_label_display.place(x=1000, y=500)
detect_flag = False
train_files = [os.path.join("/home/tung/Downloads/mx/fc-cnn-assignment/images/images", file) for file in train_df.image]
def detect_objects():
    global file_path
    global detect_flag
    global pred_labels
    if detect_flag:
        frame = cv2.imread(f'{file_path}')
        image= cv2.resize(frame,(224,224))
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=0)
        pred = loaded_model.predict(image)
        pred_labels = np.argmax(pred, axis=1)
        if train_df.label[train_files.index(f'{file_path}')]== pred_labels:
            real_label_display.config(text="True Classification.")
        img = cv2_to_pil(frame)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        # Update pred_label_display with prediction result
        pred_label_display.config(text=f"Prediction: {pred_labels[0]}")
    window.after(1, detect_objects)

# Buttons for Start, Stop, Open Folder, and Quit
def start_detection():
    global detect_flag
    detect_flag = True

def stop_detection():
    global detect_flag
    detect_flag = False

def quit_app():
    window.destroy()


start_button = tk.Button(window, text="Start", bg="green", fg="white", font=("Times New Roman", 18), command=start_detection)  # Nút start có màu xanh
start_button.place(relx=0.1, rely=0.9, relwidth=0.15, relheight=0.1)

stop_button = tk.Button(window, text="Stop", bg="red", fg="white", font=("Times New Roman", 18), command=stop_detection)  # Nút stop có màu đỏ
stop_button.place(relx=0.3, rely=0.9, relwidth=0.15, relheight=0.1)

open_button = tk.Button(window, text="Open Folder", bg="blue", fg="white", font=("Times New Roman", 18), command=open_folder)  # Nút mở thư mục có màu xanh dương
open_button.place(relx=0.5, rely=0.9, relwidth=0.2, relheight=0.1)

quit_button = tk.Button(window, text="Quit", bg="black", fg="white", font=("Times New Roman", 18), command=quit_app)  # Nút quit có màu đen
quit_button.place(relx=0.8, rely=0.9, relwidth=0.15, relheight=0.1)

window.after(1, detect_objects)  
window.mainloop()
