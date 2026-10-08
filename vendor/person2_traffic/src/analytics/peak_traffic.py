import time


class PeakTrafficAnalyzer:
    """
    Tracks the maximum number of vehicles
    observed during the current session.
    """

    def __init__(self):
        self.peak_vehicles = 0
        self.peak_time = None
        self.peak_timestamp = None

    def update(self, vehicle_count):
        """
        Update the peak vehicle count.
        """

        if vehicle_count > self.peak_vehicles:

            self.peak_vehicles = vehicle_count

            self.peak_timestamp = time.time()

            self.peak_time = time.strftime(
                "%H:%M:%S",
                time.localtime(
                    self.peak_timestamp
                )
            )

    def get_peak_vehicles(self):
        """
        Return maximum number of vehicles
        seen during the session.
        """

        return self.peak_vehicles

    def get_peak_time(self):
        """
        Return time when peak traffic occurred.
        """

        if self.peak_time is None:
            return "--:--:--"

        return self.peak_time

    def get_result(self):
        """
        Return peak traffic information.
        """

        return {
            "peak_vehicles": self.peak_vehicles,
            "peak_time": self.get_peak_time()
        }

    def reset(self):
        """
        Reset peak traffic statistics.
        """

        self.peak_vehicles = 0
        self.peak_time = None
        self.peak_timestamp = None