#  Copyright © 2024 Thomas Edward Burns
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#  permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#  Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
            if self.current_interval % self.intervals_before_refill != 0 and self.token_rate_limit_exceeded_count > 0:
                self.add_to_request_limit_queue = 0 # if we are not at the end of the interval, don't add any tickets
            elif ((self.current_interval % self.intervals_before_refill) == 0) and self.token_rate_limit_exceeded_count > 0:
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
        try:
            self.timer.start()
        except RuntimeError as re:
            pass #not great...but this happens as we are exiting the program

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