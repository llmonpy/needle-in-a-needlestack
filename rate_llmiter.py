import concurrent
import os
import threading
import time
from queue import Queue, Empty

MINUTE_TIME_WINDOW = 60
SECOND_TIME_WINDOW = 1


'''
This function takes in the number of requests per minute and returns the number of requests per second.  This spreads
the requests out over the minute time window rather than allowing them all to be made at once.
'''
def spread_requests(requests_per_minute):
    requests_per_second = round(requests_per_minute / 60)
    return requests_per_second, SECOND_TIME_WINDOW, 60


class RateLlmiter:
    def __init__(self, request_limit, time_window, intervals_before_refill=1, timeout=None):
        self.request_limit = request_limit
        self.time_window = time_window
        self.intervals_before_refill = intervals_before_refill
        self.current_interval = 0
        self.timeout = timeout
        self.request_limit_queue = Queue()
        self.token_rate_limit_exceeded_queue = Queue()
        self.token_rate_limit_exceeded_count = 0
        self.token_rate_limit_exceeded_lock = threading.Lock()
        self.add_tickets()
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tickets)
        self.timer.start()

    def add_tickets(self):
        add_to_request_limit_queue = self.request_limit
        add_too_rate_limit_exceeded_queue = 0
        with self.token_rate_limit_exceeded_lock:
            self.current_interval += 1
            if ((self.current_interval % self.intervals_before_refill) == 0) and self.token_rate_limit_exceeded_count > 0:
                if self.token_rate_limit_exceeded_count < self.request_limit:
                    add_too_rate_limit_exceeded_queue = self.token_rate_limit_exceeded_count
                    add_to_request_limit_queue = self.request_limit - self.token_rate_limit_exceeded_count
                    self.token_rate_limit_exceeded_count = 0
                else:
                    self.token_rate_limit_exceeded_count -= self.request_limit
                    add_too_rate_limit_exceeded_queue = self.request_limit
                    add_to_request_limit_queue = 0
        for _ in range(add_too_rate_limit_exceeded_queue):
            self.token_rate_limit_exceeded_queue.put("ticket")
        while not self.request_limit_queue.empty():
            self.request_limit_queue.get_nowait()
        for _ in range(add_to_request_limit_queue):
            self.request_limit_queue.put("ticket")
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tickets)
        self.timer.start()

    def get_ticket(self):
        try:
            result = self.request_limit_queue.get(timeout=self.timeout)
        except Empty as empty:
            raise empty
        return result

    def wait_for_ticket_after_rate_limit_exceeded(self):
        with self.token_rate_limit_exceeded_lock:
            self.token_rate_limit_exceeded_count += 1
        try:
            result = self.token_rate_limit_exceeded_queue.get(timeout=self.timeout)
        except Empty as empty:
            raise empty
        return result