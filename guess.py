import pickle
import pygame
from scale_data import ScaleData
import numpy as np
import pandas as pd
import pickle
from tkinter import messagebox
from tkinter import *
import sys


def mouse_input():
    """
    Getting the mouse input
    """
    is_clicked = pygame.mouse.get_pressed(num_buttons=5)
    mouse_pos = pygame.mouse.get_pos()
    if is_clicked[0]:
        return mouse_pos


class MnistGui:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Number Guesser")
        self.svm = pickle.load(open('svm_model.pkl', 'rb'))
        self.width = self.height = 500
        self.bg_color = (255, 255, 255)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.cursor_pos = []
        self.thickness_x = self.thickness_y = 50
        self.root = Tk()
        self.root.eval(f"tk::PlaceWindow {self.root.winfo_toplevel()} center")
        self.root.withdraw()

    def run(self):
        """
        Running the whole main loop
        """
        while True:
            self.mouse_pos = mouse_input()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    try:
                        scaled_data = np.array([ScaleData(self.cursor_pos, thick_x=self.thickness_x,
                                                             thick_y=self.thickness_y).scale_img_data()]) 
                        print(scaled_data.shape)
                    
                        print("Loading the model...")
                        n_sample, nx, ny = scaled_data.shape
                        scaled_data = scaled_data.reshape((n_sample, nx * ny))
                        column_names = [f"pixel_{i}" for i in range(scaled_data.shape[1])]
                        df = pd.DataFrame(scaled_data, columns=column_names)
                        print(df)
                        model = self.svm
                        prediction = model.predict(df)
                        messagebox.showinfo("Prediction", f"I think this number is {prediction[0]}")
                    except Exception:
                        messagebox.showerror("Error", "Number cannot be matched. Please try again")
                    self.root.quit()
                    self.cursor_pos.clear()
            if self.mouse_pos is not None:
                self.cursor_pos.append((self.mouse_pos[0], self.mouse_pos[1]))
            self.draw()
            self.screen.fill(self.bg_color)

    def draw(self):
        """
        Fill the pixel on the screen
        """
        try:
            color = (0, 0, 0)
            for cursor_pos in self.cursor_pos:
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(cursor_pos[0], cursor_pos[1], self.thickness_x, self.thickness_y))
            pygame.display.update()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    MnistGui().run()