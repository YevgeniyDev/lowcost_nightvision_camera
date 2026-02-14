#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

// --- GLOBAL SHARED RESOURCES ---
cv::Mat globalFrame;
std::mutex frameMutex;
std::atomic<bool> isRunning(true);

// ================== MJPEG CLIENT HANDLER ==================
void handleClient(int clientSock) {
    // Read and ignore HTTP request (we don't care about the URL/path)
    char buffer[1024];
    int n = recv(clientSock, buffer, sizeof(buffer)-1, 0);
    if (n <= 0) {
        close(clientSock);
        return;
    }
    buffer[n] = '\0';

    std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Server: NightVisionMJPEG\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=boundarydonotcross\r\n\r\n";

    if (send(clientSock, header.c_str(), header.length(), MSG_NOSIGNAL) < 0) {
        close(clientSock);
        return;
    }

    std::cout << "[MJPEG] Client " << clientSock << " connected\n";

    while (isRunning) {
        cv::Mat img;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (globalFrame.empty()) {
                // no frame yet
                // small sleep outside of critical section
                ;
            } else {
                img = globalFrame.clone();
            }
        }

        if (img.empty()) {
            usleep(10000);
            continue;
        }

        std::vector<uchar> bufferJpeg;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 50};
        cv::imencode(".jpg", img, bufferJpeg, params);

        std::string boundary = "--boundarydonotcross\r\n";
        std::string contentType = "Content-Type: image/jpeg\r\n";
        std::string contentLength =
            "Content-Length: " + std::to_string(bufferJpeg.size()) + "\r\n\r\n";

        if (send(clientSock, boundary.c_str(), boundary.length(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, contentType.c_str(), contentType.length(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, contentLength.c_str(), contentLength.length(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, bufferJpeg.data(), bufferJpeg.size(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, "\r\n", 2, MSG_NOSIGNAL) < 0) break;

        // ~30 FPS
        usleep(33000);
    }

    std::cout << "[MJPEG] Client " << clientSock << " disconnected\n";
    close(clientSock);
}

// ================== MJPEG STREAMING SERVER ==================
void mjpegServer(int port) {
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket failed");
        return;
    }

    if (setsockopt(server_fd, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT,
                   &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return;
    }

    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);  // listen on all interfaces
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        close(server_fd);
        return;
    }

    if (listen(server_fd, 10) < 0) {
        perror("Listen");
        close(server_fd);
        return;
    }

    std::cout << "🌐 MJPEG Server running on port " << port
              << " (http://<jetson-ip>:" << port << "/)\n";

    while (isRunning) {
        int new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (new_socket < 0) {
            if (!isRunning) break;
            continue;
        }

        // Each client in its own thread
        std::thread t(handleClient, new_socket);
        t.detach();
    }

    close(server_fd);
    std::cout << "[MJPEG] Server stopped.\n";
}

// ================== NIGHT VISION LOGIC CLASS ==================
class NightVisionCam {
private:
    cv::VideoCapture cap;
    cv::Ptr<cv::CLAHE> clahe;
    cv::Mat sharpen_kernel;

    // CLAHE Clip Limit
    const double CLAHE_CLIP_LIMIT = 4.0;
    const cv::Size CLAHE_TILE_GRID = cv::Size(8, 8);

public:
    NightVisionCam(int src) {
        // Prefer V4L2 backend (more stable on Jetson)
        cap.open(src, cv::CAP_V4L2);

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            exit(-1);
        }

        // Initialize CLAHE
        clahe = cv::createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID);

        // --- SHARPENING KERNEL (same as you) ---
        sharpen_kernel = (cv::Mat_<float>(3, 3) <<
                           0, -1,  0,
                          -1,  5, -1,
                           0, -1,  0);
    }

    void run() {
        cv::Mat frame, gray, equalized, denoised, sharp, boosted;

        std::cout << "SUCCESS: Night Vision (Clean Grayscale) Started!" << std::endl;

        while (isRunning) {
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "[CAM] Empty frame, retrying...\n";
                usleep(10000);
                continue;
            }

            // 1. Rotate
            cv::rotate(frame, frame, cv::ROTATE_180);

            // 2. Grayscale
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // 3. CLAHE (Contrast)
            clahe->apply(gray, equalized);

            // 4. Denoise
            cv::GaussianBlur(equalized, denoised, cv::Size(3, 3), 0);

            // 5. Sharpen (Moderate)
            cv::filter2D(denoised, sharp, -1, sharpen_kernel);

            // 6. Brightness Boost (+25%)
            // Alpha = 1.25 (gain), Beta = 0 (offset)
            cv::convertScaleAbs(sharp, boosted, 1.25, 0);

            // 7. Update Stream (single-channel grayscale, JPEG handles it)
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                boosted.copyTo(globalFrame);
            }
        }
    }
};

int main() {
    // Start MJPEG server in background
    std::thread serverThread(mjpegServer, 8080);

    // Start camera + night vision processing (blocking)
    NightVisionCam nv(0);
    nv.run();

    // If run() ever exits:
    isRunning = false;
    if (serverThread.joinable())
        serverThread.join();

    return 0;
}
