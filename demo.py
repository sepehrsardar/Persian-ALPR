"""Run: python demo.py. Choose an image; no training is started."""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import torch
from PIL import ImageTk

from predict import ROOT, draw_plates, load_models, read_plates, save_predictions


def main():
    window = tk.Tk()
    window.title("Persian plate reader")
    window.geometry("1000x780")
    status = tk.StringVar(value="Choose a photo to read its plates.")
    tk.Label(window, textvariable=status).pack(pady=8)
    # Keep the experimental correction opt-in; it previously reduced overall validation accuracy.
    rectify = tk.BooleanVar(value=False)
    tk.Checkbutton(window, text="Perspective correction (experimental, off by default)", variable=rectify).pack()
    frame = tk.Frame(window)
    frame.pack(fill="both", expand=True)
    # A scrollable canvas keeps all detected plate panels accessible.
    canvas = tk.Canvas(frame, background="#eef2f6")
    scrollbar = tk.Scrollbar(frame, command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(fill="both", expand=True)
    text = tk.StringVar()
    tk.Label(window, textvariable=text, font=("Segoe UI", 16)).pack(pady=8)
    # The worker sends results through a queue; Tk widgets are updated only on the UI thread.
    events = queue.Queue()
    models = None

    def process(path, use_rectification):
        nonlocal models
        try:
            # Load weights on the first request and reuse them for subsequent images.
            if models is None:
                models = load_models()
            device = "0" if torch.cuda.is_available() else "cpu"
            plates = read_plates(cv2.imread(path), *models, device=device, rectify=use_rectification)
            save_predictions(plates, os.path.join(ROOT, "outputs", "demo"))
            events.put(("success", plates))
        except Exception as error:
            # Report worker errors on the UI thread; never disguise a failure as an empty result.
            events.put(("error", f"{type(error).__name__}: {error}"))

    def choose_image():
        path = filedialog.askopenfilename(filetypes=[("Photos", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        if path:
            # Allow only one active prediction so shared models and output files are not used concurrently.
            button.configure(state="disabled")
            status.set("Reading plates...")
            # Run inference outside the UI thread to keep the window responsive.
            threading.Thread(target=process, args=(path, rectify.get()), daemon=True).start()

    def show_result():
        try:
            kind, result = events.get_nowait()
        except queue.Empty:
            window.after(100, show_result)
            return
        button.configure(state="normal")
        if kind == "error":
            messagebox.showerror("Prediction failed", result)
            status.set("Prediction failed")
        else:
            # The shared renderer draws compact preview boxes without changing model detections.
            picture = draw_plates(result)
            # Keep a Python reference so Tk does not discard the displayed image.
            canvas.image = ImageTk.PhotoImage(picture)
            canvas.delete("all")
            canvas.create_image(0, 0, image=canvas.image, anchor="nw")
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.yview_moveto(0)
            text.set("   |   ".join(plate["text"] or "(no characters)" for plate in result) if result else "No plate found")
            status.set(f"{len(result)} plate(s). PNG and CSV saved in outputs/demo.")
        window.after(100, show_result)

    button = tk.Button(window, text="Choose photo", command=choose_image)
    button.pack(pady=10)
    # Poll the queue using the Tk event loop rather than accessing widgets from the worker.
    window.after(100, show_result)
    window.mainloop()


if __name__ == "__main__":
    main()
