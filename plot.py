@staticmethod
    def plot_metric(ax, data, line_info=None, ylim=None):
        """
        Helper function to plot a GPU metric on a given axis.

        Parameters:
        - ax: The axis on which to plot.
        - data: A tuple containing (timestamps, y_data, title, ylabel, xlabel).
        - line_info: A tuple containing (yline, yline_label) for the horizontal line (optional).
        """
        timestamps, y_data, title, ylabel, xlabel = data

        for i, gpu_data in enumerate(y_data):
            ax.plot(timestamps, gpu_data, label=f'GPU {i}', marker='*')

        if line_info:
            yline, yline_label = line_info
            ax.axhline(y=yline, color='r', linestyle='--', label=yline_label)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='bold')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Set y-axis limits if provided
        if ylim:
            ax.set_ylim(ylim)

    def plot_metrics(self):
        """
        Plot and save the GPU metrics to a file.
        """
        timestamps = self._time_series_data['timestamp']

        # Prepare data for plotting
        power_data = [[p[i] for p in self._time_series_data['power']] for i in range(self.device_count)]
        util_data = [[u[i] for u in self._time_series_data['util']] for i in range(self.device_count)]
        temp_data = [[t[i] for t in self._time_series_data['temp']] for i in range(self.device_count)]
        mem_data = [[m[i] for m in self._time_series_data['mem']] for i in range(self.device_count)]

        # Plot 2x2 grid of power, utilization, temperature, and memory
        fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
        fig1.tight_layout(pad=10.0)

        # Plot each metric using the helper function
        self.plot_metric(
            axes[0, 0],
            (timestamps,
             power_data,
             f'GPU Power Usage, Total Energy: {self._stats["total_energy"]:.3g}kWh',
             'Power (W)', None),
            (self._stats["max_power_limit"], 'Power Limit')
        )
        self.plot_metric(
            axes[0, 1],
            (timestamps, util_data, 'GPU Utilization', 'Utilization (%)', None),
            ylim=(0, 100)  # Set y-axis limits for utilization
        )
        self.plot_metric(
            axes[1, 0],
            (timestamps, temp_data, 'GPU Temperature', 'Temperature (C)', 'Timestamp')
        )
        self.plot_metric(
            axes[1, 1],
            (timestamps, mem_data, 'GPU Memory Usage', 'Memory (MiB)', 'Timestamp'),
            (self._stats["total_mem"], 'Total Memory')
        )

        # Save the 2x2 grid plot
        plt.savefig(METRIC_PLOT_PATH, bbox_inches='tight')
        plt.close(fig1)