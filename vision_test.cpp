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

// --- MJPEG STREAMING SERVER (Unchanged) ---
void mjpegServer(int port) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        return;
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        return;
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        return;
    }
    if (listen(server_fd, 3) < 0) {
        perror("Listen");
        return;
    }

    std::cout << "🌐 MJPEG Server running on port " << port << std::endl;

    while (isRunning) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            continue;
        }
        std::string header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=boundarydonotcross\r\nConnection: close\r\n\r\n";
        send(new_socket, header.c_str(), header.length(), 0);

        while (isRunning) {
            cv::Mat img;
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                if (globalFrame.empty()) { usleep(10000); continue; }
                img = globalFrame.clone();
            }

            std::vector<uchar> buffer;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 50}; 
            cv::imencode(".jpg", img, buffer, params);

            std::string boundary = "--boundarydonotcross\r\n";
            std::string contentType = "Content-Type: image/jpeg\r\n";
            std::string contentLength = "Content-Length: " + std::to_string(buffer.size()) + "\r\n\r\n";
            
            if (send(new_socket, boundary.c_str(), boundary.length(), MSG_NOSIGNAL) < 0) break;
            if (send(new_socket, contentType.c_str(), contentType.length(), MSG_NOSIGNAL) < 0) break;
            if (send(new_socket, contentLength.c_str(), contentLength.length(), MSG_NOSIGNAL) < 0) break;
            if (send(new_socket, buffer.data(), buffer.size(), MSG_NOSIGNAL) < 0) break;
            if (send(new_socket, "\r\n", 2, MSG_NOSIGNAL) < 0) break;

            usleep(33000); 
        }
        close(new_socket);
    }
    close(server_fd);
}

// --- VISUALIZATION HELPER ---
// Adds a label to an image
void addLabel(cv::Mat& img, std::string text) {
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols, 25), cv::Scalar(0), -1); // Black banner
    cv::putText(img, text, cv::Point(5, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1);
}

class NightVisionDemo {
private:
    cv::VideoCapture cap;
    cv::Ptr<cv::CLAHE> clahe;
    cv::Mat sharpen_kernel;
    
    // Constants
    const double CLAHE_CLIP_LIMIT = 4.0; 
    const cv::Size CLAHE_TILE_GRID = cv::Size(8, 8);

public:
    NightVisionDemo(int src) {
        cap.open(src, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        clahe = cv::createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID);

        // Moderate Sharpening Kernel
        sharpen_kernel = (cv::Mat_<float>(3, 3) << 
                           0, -1,  0, 
                          -1,  5, -1, 
                           0, -1,  0);
    }

    void run() {
        cv::Mat frame, gray, equalized, denoised, sharp, boosted;
        cv::Mat grid, topRow, bottomRow;
        
        // Small temporary mats for resizing
        cv::Mat s_gray, s_clahe, s_denoise, s_sharp, s_boost, s_final;

        std::cout << "SUCCESS: Pipeline Visualization Started!" << std::endl;

        while (isRunning) {
            cap >> frame;
            if (frame.empty()) break;

            // --- THE PIPELINE ---
            
            // 1. Raw Input (Rotate & Gray)
            cv::rotate(frame, frame, cv::ROTATE_180);
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // 2. CLAHE
            clahe->apply(gray, equalized);

            // 3. Denoise
            cv::GaussianBlur(equalized, denoised, cv::Size(3, 3), 0);

            // 4. Sharpen
            cv::filter2D(denoised, sharp, -1, sharpen_kernel);

            // 5. Boost (+25%)
            cv::convertScaleAbs(sharp, boosted, 1.25, 0);

            // --- CREATE THE GRID FOR DEMONSTRATION ---
            
            // Resize all images to smaller size (e.g., 320x240) to fit on screen
            cv::Size smallSize(640, 480);
            cv::resize(gray, s_gray, smallSize);
            cv::resize(equalized, s_clahe, smallSize);
            cv::resize(denoised, s_denoise, smallSize);
            cv::resize(sharp, s_sharp, smallSize);
            cv::resize(boosted, s_boost, smallSize);
            
            // Make a copy of boosted for the "Final" slot
            s_final = s_boost.clone();

            // Add Labels
            addLabel(s_gray,   "1. Raw Grayscale");
            addLabel(s_clahe,  "2. CLAHE (Contrast)");
            addLabel(s_denoise,"3. Gaussian Denoise");
            addLabel(s_sharp,  "4. Laplacian Sharpen");
            addLabel(s_boost,  "5. Brightness +25%");
            addLabel(s_final,  "== FINAL RESULT ==");

            // Stitch Top Row
            std::vector<cv::Mat> top = {s_gray, s_clahe, s_denoise};
            cv::hconcat(top, topRow);

            // Stitch Bottom Row
            std::vector<cv::Mat> bottom = {s_sharp, s_boost, s_final};
            cv::hconcat(bottom, bottomRow);

            // Stitch Top and Bottom
            cv::vconcat(topRow, bottomRow, grid);

            // Update Stream
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                grid.copyTo(globalFrame);
            }
        }
    }
};

int main() {
    std::thread serverThread(mjpegServer, 8080);
    serverThread.detach(); 
    
    NightVisionDemo demo(0);
    demo.run();
    
    return 0;
}
