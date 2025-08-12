class TimeKeeper:
    def __init__(self, task_durations):
        """
        Initialize the TimeKeeper with a dictionary of task durations.
        task_durations: dict {task_name: duration}
        """
        self.task_durations = task_durations  # Task durations by key (name)
        self.busy = {key: 0 for key in task_durations}  # Busy status of tasks
        self.completion_times = {key: 0 for key in task_durations}  # Completion times
        self.current_time = 0
        self.pending_tasks = []  # List of task keys to activate later

    def add_tasks(self, task_keys):
        """
        Add tasks (by key) to the list of pending tasks to be activated later.
        task_keys: list of task keys to add
        """
        self.pending_tasks.append(task_keys)

    def activate_pending_tasks(self):
        """
        Activate all pending tasks at the current time.
        """
        for key in self.pending_tasks:
            if self.busy[key] == 0:
                self.busy[key] = 1
                self.completion_times[key] = self.current_time + self.task_durations[key]
            else:
                return -1

        self.pending_tasks = []
        return True

    def get_next_time(self):
        """
        Simulate until the next task completion.
        Update the current time and busy statuses.
        Return the current time and a copy of the busy statuses.
        """
        next_times = [self.completion_times[key] for key in self.busy if self.busy[key]]
        if not next_times:
            return self.current_time, self.busy.copy()

        # Advance current time to the next event
        next_completion_time = min(next_times)
        self.current_time = next_completion_time

        # Identify and process tasks that complete at this time
        completed_tasks = [key for key in self.busy
                           if self.busy[key] and self.completion_times[key] == self.current_time]
        for key in completed_tasks:
            self.busy[key] = 0
            self.completion_times[key] = 0

        return self.current_time, self.busy.copy()

    def get_status(self):
        """
        Return the current time and a copy of the busy statuses.
        """
        return self.current_time, self.busy.copy()
