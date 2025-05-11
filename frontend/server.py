from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import socket
import sys
import logging


class EnvVariableHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Set default page to index.html
        if self.path == "/" or self.path == "":
            self.path = "/index.html"

        try:
            # Try to open the requested file
            file_to_open = "." + self.path
            with open(file_to_open, "r", encoding="utf-8") as file:
                content = file.read()

                # Get environment variables with default values
                API_URL = os.getenv("API_URL", "http://localhost:8080/predict")
                logging.info(f"API_URL: {API_URL}")

                feedback_endpoint = os.getenv(
                    "FEEDBACK_ENDPOINT", "http://localhost:8080/feedback"
                )
                timeout_ms = os.getenv("TIMEOUT_MS", "5000")

                # Replace placeholders in HTML
                content = content.replace("{{API_ENDPOINT}}", API_URL)
                content = content.replace("{{FEEDBACK_ENDPOINT}}", feedback_endpoint)

                # Process other environment variables
                for var_name, var_value in os.environ.items():
                    placeholder = "{{" + var_name + "}}"
                    if placeholder in content:
                        content = content.replace(placeholder, str(var_value))

                # Send successful response
                self.send_response(200)
                # Set correct Content-type based on file extension
                if self.path.endswith(".html"):
                    self.send_header("Content-type", "text/html")
                elif self.path.endswith(".css"):
                    self.send_header("Content-type", "text/css")
                elif self.path.endswith(".js"):
                    self.send_header("Content-type", "application/javascript")
                else:
                    self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))

        except FileNotFoundError:
            # File not found, return 404 error
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"404 - File Not Found")
        except Exception as e:
            # Other errors
            self.send_response(500)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"500 - Server Error: {str(e)}".encode("utf-8"))


def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"  # Return localhost if unable to get IP


def run_server(port=8000):
    """Run HTTP server"""
    print(f"Starting server on port {port}...")
    server_address = ("", port)
    httpd = HTTPServer(server_address, EnvVariableHandler)
    local_ip = get_local_ip()

    # Print current environment variable settings

    print(f"\nServer started:")
    print(f"- Local access: http://localhost:{port}")
    print(f"- Network access: http://{local_ip}:{port}")
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        httpd.server_close()
        sys.exit(0)


if __name__ == "__main__":
    # Use port from command line argument if provided
    port = 8000
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            print(f"Using default port: {port}")

    run_server(port)
