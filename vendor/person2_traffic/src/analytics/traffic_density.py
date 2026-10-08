class TrafficDensity:

    def __init__(self, maximum_vehicles=20):
        """
        maximum_vehicles:
            Approximate number of vehicles that represent
            100% traffic density for the current camera view.
        """

        self.maximum_vehicles = maximum_vehicles

    def calculate(self, vehicle_count):
        """
        Calculate normalized traffic density.

        Returns:
            Density as a value from 0.0 to 1.0
        """

        if self.maximum_vehicles <= 0:
            return 0.0

        density = vehicle_count / self.maximum_vehicles

        # Keep density between 0 and 1
        density = max(0.0, min(1.0, density))

        return density

    def percentage(self, vehicle_count):
        """
        Return density as percentage.
        """

        density = self.calculate(vehicle_count)

        return density * 100

    def traffic_level(self, vehicle_count):
        """
        Convert density into traffic level.
        """

        density = self.calculate(vehicle_count)

        if density < 0.25:
            return "LOW"

        elif density < 0.50:
            return "MEDIUM"

        elif density < 0.75:
            return "HIGH"

        else:
            return "CRITICAL"