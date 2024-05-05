import concurrent
import os
import threading
import time
from queue import Queue, Empty


class RateLlmiter:
    def __init__(self, request_limit, time_window, timeout=None):
        self.request_limit = request_limit
        self.time_window = time_window
        self.timeout = timeout
        self.request_limit_queue = Queue()
        self.token_rate_limit_exceeded_queue = Queue()
        self.token_rate_limit_exceeded_count = 0
        self.token_rate_limit_exceeded_lock = threading.Lock()
        self.add_tokens()
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tokens)
        self.timer.start()

    def add_tokens(self):
        add_to_request_limit_queue = self.request_limit
        add_too_rate_limit_exceeded_queue = 0
        with self.token_rate_limit_exceeded_lock:
            if self.token_rate_limit_exceeded_count > 0:
                if self.token_rate_limit_exceeded_count < self.request_limit:
                    add_too_rate_limit_exceeded_queue = self.token_rate_limit_exceeded_count
                    add_to_request_limit_queue = self.request_limit - self.token_rate_limit_exceeded_count
                    self.token_rate_limit_exceeded_count = 0
                else:
                    self.token_rate_limit_exceeded_count -= self.request_limit
                    add_too_rate_limit_exceeded_queue = self.request_limit
                    add_to_request_limit_queue = 0
        for _ in range(add_too_rate_limit_exceeded_queue):
            self.token_rate_limit_exceeded_queue.put("token")
        while not self.request_limit_queue.empty():
            self.request_limit_queue.get_nowait()
        for _ in range(add_to_request_limit_queue):
            self.request_limit_queue.put("token")
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tokens)
        self.timer.start()

    def get_token(self):
        try:
            result = self.request_limit_queue.get(timeout=self.timeout)
        except Empty as empty:
            raise empty
        return result

    def wait_for_token_after_rate_limit_exceeded(self):
        while not self.request_limit_queue.empty():
            self.request_limit_queue.get_nowait()
        with self.token_rate_limit_exceeded_lock:
            self.token_rate_limit_exceeded_count += 1
        try:
            result = self.token_rate_limit_exceeded_queue.get(timeout=self.timeout)
        except Empty as empty:
            raise empty
        return result