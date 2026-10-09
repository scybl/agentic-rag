"""TLS 握手在线程内执行；握手和读取空闲有超时，连接线程数有上限。"""

import ssl
import threading
from http.server import ThreadingHTTPServer


class BoundedTLSHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, ssl_context, connection_timeout=8, max_connections=32):
        if connection_timeout <= 0 or max_connections < 1:
            raise ValueError("连接期限和连接上限必须为正数")
        self.ssl_context = ssl_context
        self.connection_timeout = connection_timeout
        self.connection_slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def get_request(self):
        raw, address = self.socket.accept()
        try:
            raw.settimeout(self.connection_timeout)
            # accept 线程绝不等待客户端发送 TLS 字节。
            return self.ssl_context.wrap_socket(raw, server_side=True, do_handshake_on_connect=False), address
        except BaseException:
            raw.close()
            raise

    def process_request(self, request, client_address):
        if not self.connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            try:
                request.do_handshake()
            except (OSError, ssl.SSLError):
                self.shutdown_request(request)
                return
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()
