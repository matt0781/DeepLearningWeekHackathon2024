import tkinter as tk
import abc
from tkinter import filedialog, PhotoImage, messagebox
import shutil
import os
import ctypes
import subprocess
import time

ctypes.windll.shcore.SetProcessDpiAwareness(True)

def openwindow(checkbox_value):
    file_path = filedialog.askopenfilename()

# Check if a file was selected
    if file_path:
    # Specify the target directory
        target_directory = "./metafiles"
    
    # Create the target directory if it doesn't exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
    
    # Extract the file name from the file path
        file_name = os.path.basename(file_path)
    
    # Construct the full path to the target file
        target_file_path = os.path.join(target_directory, file_name)
    
    # Copy the file to the target directory 
        shutil.copy(file_path, target_file_path)
    
    # Optionally, display a message to the user
        print(f"File {file_name} uploaded to {target_directory}")

        if any(os.scandir(target_directory)):
            
            messagebox.showinfo("Running", "The process is running. Please wait...\nFinished message will pop up once the process is done.")
            # If there are files, run another Python script
            
        
            
            result = subprocess.run(["python", "./oneshot_script.py", str(checkbox_value)], check=True)

            
            if result.returncode == 0:
                print("Script executed successfully. Done.")
                messagebox.showinfo("Job Finished", "Video clips filtering is done!\nOpen your results folder in the same directory as this script.")
            else:
                messagebox.showinfo("Job Failed", "Video clips filtering failed!")
def main():
    # Create the main window
    root = tk.Tk()
    root.title("Filtering of video data")
    root.geometry("300x200")
    # Set dark mode colors
    root.configure(bg='#282828') # Dark background for the window

    # Configure the grid to have weighted columns and rows
    for i in range(21):
        root.grid_rowconfigure(i, weight=1)
        root.grid_columnconfigure(i, weight=1)
    
    checkbox_var = tk.BooleanVar(value=True)  # Set initial value to True
    checkbox = tk.Checkbutton(root, text="Check for diverse selection of top clips", variable=checkbox_var, bg='#282828', fg='white', font=("Helvetica", 16), selectcolor='#282828')
    checkbox.grid(row=14, column=10, sticky='nsew')

    image = PhotoImage(file="./frontendAssets/banner.png")
    # Create a label to display the image
    image_label = tk.Label(root, image=image, bg='#282828')
    image_label.image = image # Keep a reference to the image
    # Place the image label above the button
    image_label.grid(row=3, column=10, sticky='nsew')

    text_label = tk.Label(root, text="Upload your Json file containing the information of the videos.\nIt will generate a Json file, with path location of choosen clips, along with the captions and scores.", bg='#282828', fg='white', font=("Helvetica", 16))
    text_label.grid(row=10, column=10, sticky='nsew')
    # Create a button
    button = tk.Button(root, text="Upload Json file", bg='#444444', fg='white', command=lambda: openwindow(checkbox_var.get()), font=("Helvetica", 16))

    # Place the button in the center of the grid
    button.grid(row=13, column=10,)

    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()