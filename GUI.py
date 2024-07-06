# Importing necessary PyQt5 modules for GUI elements, and matplotlib for plotting within the PyQt5 framework.
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from processing import main


"""
PlotCanvas:
A custom QWidget to display matplotlib plots. Inherits from FigureCanvasQTAgg to integrate matplotlib figures
with PyQt5.
This class is central to visualizing the density charts for stock movements.
Currently working on more nuanced error handling.
"""


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(15, 9))
        super().__init__(fig)
        self.setParent(parent)
        plt.tight_layout()

    def plot(self, ticker_symbol, days):
        main(ticker_symbol, days, self.ax)


"""
MainWindow:
Defines the main window for the application. It sets up the layout and widgets, including
labels, text inputs, and buttons.
This class serves as the main interface for the user, facilitating input for the stock ticker and days, and
triggering plots based on user interactions.
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Move Density Chart")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.ticker_label = QLabel("Ticker Symbol:")
        layout.addWidget(self.ticker_label)
        self.ticker_input = QLineEdit()
        layout.addWidget(self.ticker_input)

        self.days_label = QLabel("Lookback Period Days (e.g., '100'):")
        layout.addWidget(self.days_label)
        self.days_input = QLineEdit()
        layout.addWidget(self.days_input)

        self.plot_button = QPushButton("Plot")
        layout.addWidget(self.plot_button)
        self.plot_button.clicked.connect(self.plot)

        self.canvas = PlotCanvas(self)
        layout.addWidget(self.canvas)

    def plot(self):
        ticker_symbol = self.ticker_input.text()
        days = int(self.days_input.text())
        self.canvas.plot(ticker_symbol, days)


"""
__main__:
The entry point when the script is run. It initializes the QApplication, sets up the main window, and starts
the event loop.
This segment orchestrates the entire application's lifecycle and user interaction, ensuring the GUI is
responsive and functional.
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
