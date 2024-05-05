import pickle
import pygame
from scale_window import ScaleWindow
import numpy as np
import pickle
from tkinter import Tk, messagebox
import sys

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
        self.scaler = pickle.load(open('standardize_model.pkl', 'rb'))
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
                        scaled_window = np.array([ScaleWindow(self.cursor_pos, thick_x=self.thickness_x, thick_y=self.thickness_y).scale_img_window()]).reshape(1, -1)
                        print(scaled_window.shape)

                        print("Loading the model...")

                        user_data_scaled = self.scaler.transform(scaled_window) # replaced the original train with the rescaled version

                        plt.imshow(user_data_scaled.reshape(28, 28), cmap='gray') # in a 28 x 28 grid
                        plt.show()

                        user_data_scaled.reshape(1, -1)

                        model = self.svm
                        prediction = model.predict(user_data_scaled)
                        messagebox.showinfo("Prediction", f"I think this number is {prediction[0]}")
                    except Exception as e:
                        print("Error:", e)
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
                pygame.draw.rect(self.screen, color, pygame.Rect(cursor_pos[0], cursor_pos[1], self.thickness_x, self.thickness_y))
            pygame.display.update()
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    MnistGui().run()