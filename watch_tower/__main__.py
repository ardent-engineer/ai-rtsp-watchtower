import sys
import yaml
import torch
import qdarkstyle
from PyQt5.QtWidgets import QApplication, QDialog

# --- Local Project Imports ---
from watch_tower.ui.dialogs import LoginDialog, WelcomePage
from watch_tower.main_window import MainWindow

def main():
    # 1. Setup Application
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # 2. Load Configuration
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: config.yaml not found. Please ensure it exists in the root directory.")
        return
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return

    # 3. Add YOLOv5 directory to system path so its modules can be imported by workers
    sys.path.insert(0, config['paths']['yolo_dir'])
    print(f"Using CUDA: {torch.cuda.is_available()}")

    # 4. Show Login Dialog
    login_dialog = LoginDialog(config)
    if login_dialog.exec_() != QDialog.Accepted or login_dialog.state == 0:
        sys.exit(0)  # Exit if login is cancelled or fails

    # 5. Show Welcome/Loading Page
    welcome_page = WelcomePage(config)
    welcome_page.show()
    app.processEvents() # Ensure the welcome page is rendered

    # 6. Initialize Main Window (this will start loading models in worker threads)
    main_window = MainWindow(state=login_dialog.state, config=config)
    
    # 7. Signal that loading is complete and the app is ready
    welcome_page.ready()
    if welcome_page.exec_() == QDialog.Accepted:
        main_window.show()
        app.exec_()
        sys.exit()
    else:
        # If user closes the welcome page, gracefully exit
        for worker in main_window.camera_workers:
            worker.stop()
            worker.wait()
        sys.exit(0)


if __name__ == "__main__":
    main()