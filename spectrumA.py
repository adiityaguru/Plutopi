# spectrumA.py
# Spectrum Analyzer for ADALM-Pluto using PyQt6 and pyqtgraph.
# Matches DPG functionality: dynamic real-time updates during sweep.
# GUI is full-screen; gain is fixed (manual mode, 70 dB) without configuration.
import sys
import numpy as np
import adi
import time
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import pyqtgraph as pg

# --- Worker for SDR processing ---
class SDRWorker(QObject):
    """
    Moves the SDR processing to a separate thread to keep the GUI responsive.
    """
    # Signal to emit new data for plotting. Emits a tuple of (current_data, peak_data)
    newData = pyqtSignal(object)
    # Signal to update the status bar
    newStatus = pyqtSignal(str)
    # Signal emitted when the worker is finished
    finished = pyqtSignal()

    def __init__(self, sdr_handler):
        super().__init__()
        self.sdr_handler = sdr_handler
        self.is_running = True
        self.is_paused = False

    def run(self):
        """
        Main worker loop with real-time updates per step.
        """
        print("SDR worker thread started.")
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            try:
                # Reset for a new sweep
                for i in range(self.sdr_handler.sweep_steps):
                    if not self.is_running or self.is_paused:
                        break
                    
                    freq_hz = self.sdr_handler.frequencies_hz[i]
                    
                    # 1. Tune SDR (with lock)
                    with self.sdr_handler.sdr_lock:
                        self.sdr_handler.sdr.rx_lo = int(freq_hz)
                    time.sleep(0.005)  # LO settling time
                    
                    # 2. Capture samples (with lock)
                    with self.sdr_handler.sdr_lock:
                        _ = self.sdr_handler.sdr.rx()  # Clear buffer
                        time.sleep(0.001)
                        rx_signal = self.sdr_handler.sdr.rx()  # Capture
                    
                    # 3. Process
                    amplitude_db = self.sdr_handler.compute_rms_power(rx_signal)
                    
                    # 4. Update shared data for plotting (thread-safe)
                    with self.sdr_handler.data_lock:
                        self.sdr_handler.plot_y_data[i] = amplitude_db
                        self.sdr_handler.plot_y_peak_data[i] = max(
                            self.sdr_handler.plot_y_peak_data[i], amplitude_db
                        )
                        # Copy for emission to avoid locking UI
                        y_data_copy = list(self.sdr_handler.plot_y_data)
                        y_peak_data_copy = list(self.sdr_handler.plot_y_peak_data)
                    
                    # 5. Emit status and incremental data update
                    status_msg = (f"Sweeping... {i+1}/{self.sdr_handler.sweep_steps} "
                                  f"({freq_hz/1e9:.3f} GHz): {amplitude_db:.2f} dB")
                    self.newStatus.emit(status_msg)
                    self.newData.emit((y_data_copy, y_peak_data_copy))
                    time.sleep(0.005)  # Small delay for ~20 FPS feel
                
                # Full sweep completed
                self.newStatus.emit("Sweep completed.")
                
            except Exception as e:
                error_msg = f"SDR Worker Error: {e}"
                self.newStatus.emit(error_msg)
                print(error_msg)
                time.sleep(1)
        print("SDR worker thread stopped.")
        self.finished.emit()

    def stop(self):
        self.is_running = False

    def toggle_pause(self, paused):
        self.is_paused = paused

# --- Main Application Window ---
class PlutoSpectrumAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlutoSDR Spectrum Analyzer (PyQt6)")
        # --- SDR Configuration (gain fixed to manual/70 dB) ---
        self.uri = "ip:192.168.3.3"
        self.sweep_start_hz = 2.0e9
        self.sweep_stop_hz = 4.0e9
        self.sweep_steps = 250
        self.frequencies_hz = np.linspace(self.sweep_start_hz, self.sweep_stop_hz, self.sweep_steps)
        self.frequencies_ghz = self.frequencies_hz / 1e9
        self.sample_rate_hz = 10.0e6
        self.rf_bw_hz = 10.0e6
        self.buffer_size = 2048
        self.gain_mode = "manual"
        self.gain_value = 70  # Fixed, no UI config
        # --- State and Data ---
        self.sdr = None
        self.sdr_thread = None
        self.sdr_worker = None
        self.plot_x_data = self.frequencies_ghz.tolist()
        self.plot_y_data = [-120.0] * self.sweep_steps
        self.plot_y_peak_data = [-120.0] * self.sweep_steps
        self.data_lock = threading.Lock()
        self.sdr_lock = threading.Lock()
        # --- Initialize SDR ---
        self.init_sdr()
        
        # --- Setup UI (no gain settings) ---
        self._setup_ui()
        
        # --- Start sweep automatically ---
        if self.sdr:
            self.start_sweep()

    def init_sdr(self):
        """Initialize the ADALM-Pluto SDR with fixed gain."""
        try:
            print("Initializing SDR...")
            self.sdr = adi.ad9361(uri=self.uri)
            self.sdr.sample_rate = int(self.sample_rate_hz)
            self.sdr.rx_lo = int(self.sweep_start_hz)
            self.sdr.rx_rf_bandwidth = int(self.rf_bw_hz)
            self.sdr.rx_buffer_size = self.buffer_size
            self.sdr.gain_control_mode_chan0 = self.gain_mode
            self.sdr.rx_hardwaregain_chan0 = self.gain_value  # Fixed
            self.sdr.rx_enabled_channels = [0]
            print("SDR Initialized.")
        except Exception as e:
            self.sdr = None
            print(f"SDR Init Error: {e}")

    def _setup_ui(self):
        """Create and arrange widgets (minimal, no gain config)."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        # --- Top Controls ---
        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Sweep")
        self.start_button.clicked.connect(self.start_sweep)
        self.start_button.setMinimumSize(150, 50)  # Touch-friendly
        self.stop_button = QPushButton("Stop Sweep")
        self.stop_button.clicked.connect(self.stop_sweep)
        self.stop_button.setMinimumSize(150, 50)
        self.pause_button = QPushButton("Pause/Resume")
        self.pause_button.setCheckable(True)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setMinimumSize(150, 50)
        self.reset_peak_button = QPushButton("Reset Peak")
        self.reset_peak_button.clicked.connect(self.reset_peak_hold)
        self.reset_peak_button.setMinimumSize(150, 50)
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.reset_peak_button)
        main_layout.addLayout(controls_layout)
        # --- Plotting Widget (fills most space) ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Frequency (GHz)')
        self.plot_widget.setLabel('left', 'Power (dB)')
        self.plot_widget.setXRange(2, 4)  # Fixed like DPG
        self.plot_widget.setYRange(-80, 30)  # Initial like DPG
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_current = self.plot_widget.plot(self.plot_x_data, self.plot_y_data, pen='y', name='Current')
        self.plot_peak = self.plot_widget.plot(self.plot_x_data, self.plot_y_peak_data, pen='r', name='Peak Hold')
        self.plot_widget.addLegend()
        
        main_layout.addWidget(self.plot_widget)
        # --- Status Bar ---
        self.statusBar().showMessage("Ready.")
        
        # Disable controls if SDR is not available
        if not self.sdr:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.reset_peak_button.setEnabled(False)

    def start_sweep(self):
        if self.sdr_thread and self.sdr_thread.isRunning():
            print("Sweep is already running.")
            return
            
        if not self.sdr:
            self.statusBar().showMessage("SDR not initialized. Cannot start sweep.")
            return
        
        self.sdr_worker = SDRWorker(self)
        self.sdr_thread = QThread()
        self.sdr_worker.moveToThread(self.sdr_thread)
        
        # Connect signals
        self.sdr_thread.started.connect(self.sdr_worker.run)
        self.sdr_worker.finished.connect(self.sdr_thread.quit)
        self.sdr_worker.finished.connect(self.sdr_worker.deleteLater)
        self.sdr_thread.finished.connect(self.sdr_thread.deleteLater)
        self.sdr_worker.newData.connect(self.update_plot)
        self.sdr_worker.newStatus.connect(self.update_status)
        
        self.sdr_thread.start()
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setChecked(False)
        self.pause_button.setText("Pause/Resume")

    def stop_sweep(self):
        if self.sdr_worker:
            self.sdr_worker.stop()
        if self.sdr_thread:
            self.sdr_thread.quit()
            self.sdr_thread.wait(2000)  # Wait up to 2s
            if self.sdr_thread.isRunning():
                print("Warning: Thread did not stop gracefully.")
                self.sdr_thread.terminate()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.statusBar().showMessage("Sweep stopped.")

    def toggle_pause(self):
        paused = self.pause_button.isChecked()
        if self.sdr_worker:
            self.sdr_worker.toggle_pause(paused)
        if paused:
            self.statusBar().showMessage("Sweep paused.")
        else:
            self.statusBar().showMessage("Sweep resumed.")

    def reset_peak_hold(self):
        with self.data_lock:
            self.plot_y_peak_data = [-120.0] * self.sweep_steps
        print("Peak hold reset.")
        # Immediate update
        self.update_plot((self.plot_y_data, self.plot_y_peak_data))

    def update_plot(self, data):
        y_data, y_peak_data = data
        self.plot_current.setData(self.plot_x_data, y_data)
        self.plot_peak.setData(self.plot_x_data, y_peak_data)
        
        # Dynamic Y-scaling like DPG
        min_power = min(y_data)
        max_power = max(y_data)
        min_range = 30  # dB
        if max_power - min_power < min_range:
            center = (max_power + min_power) / 2
            max_power = center + min_range / 2
            min_power = center - min_range / 2
        padding = 10
        self.plot_widget.setYRange(min_power - padding, max_power + padding)

    def update_status(self, status):
        self.statusBar().showMessage(status)

    def compute_rms_power(self, rx_signal):
        if rx_signal is None or len(rx_signal) == 0:
            return -120.0
        try:
            instantaneous_power = np.abs(rx_signal)**2
            avg_power = np.mean(instantaneous_power)
            power_db = 10 * np.log10(avg_power + 1e-12)
            return power_db
        except Exception as e:
            print(f"Error in compute_rms_power: {e}")
            return -120.0

    def closeEvent(self, event):
        """Ensure threads are cleaned up on window close."""
        self.stop_sweep()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlutoSpectrumAnalyzer()
    window.showFullScreen()  # Full screen mode
    sys.exit(app.exec())
