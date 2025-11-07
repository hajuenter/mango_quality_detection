from threading import Lock


class TriggerService:
    def __init__(self):
        self.should_capture = False
        self.lock = Lock()

    def set_trigger(self, value: bool):
        """Set trigger dari ESP32"""
        with self.lock:
            self.should_capture = value
            print(f"🔔 Trigger diset ke: {value}")

    def get_and_reset_trigger(self):
        """Ambil status trigger dan reset ke False"""
        with self.lock:
            status = self.should_capture
            self.should_capture = False
            return status

    def check_trigger(self):
        """Cek trigger tanpa reset"""
        with self.lock:
            return self.should_capture


# Singleton instance
trigger_service = TriggerService()
