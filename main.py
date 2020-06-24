import sys

from PyQt5.QtWidgets import QApplication
from keystroke_dynamics import KeystrokeDynamics


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = KeystrokeDynamics()

    sys.exit(app.exec_())