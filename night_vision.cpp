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
#include <chrono>

// --- GLOBAL SHARED RESOURCES ---
cv::Mat globalFrame;
std::mutex frameMutex;
std::atomic<bool> isRunning(true);

// ================== MJPEG CLIENT HANDLER ==================
void handleClient(int clientSock) {
    // Read and ignore HTTP request
    char buffer[1024];
    int n = recv(clientSock, buffer, sizeof(buffer) - 1, 0);
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
            if (!globalFrame.empty()) img = globalFrame.clone();
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

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return;
    }

    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
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

        std::thread t(handleClient, new_socket);
        t.detach();
    }

    close(server_fd);
    std::cout << "[MJPEG] Server stopped.\n";
}

// ================== CLI OPTIONS ==================
struct Options {
    bool recordMode = false;
    std::string recordPath = "";
    int recordSeconds = 20;           // default duration
    bool streamEnhancedWhileRecord = false; // if false, stream raw grayscale while recording
    int port = 8080;
    int camIndex = 0;
};

static void printUsage(const char* prog) {
    std::cout <<
        "Usage:\n"
        "  " << prog << "                  # enhance + stream\n"
        "  " << prog << " --record out.avi  # record RAW dataset + stream preview\n"
        "Options:\n"
        "  --record <path>         Record raw (rotated) BGR frames to AVI (MJPG codec)\n"
        "  --seconds <N>           Recording duration in seconds (default: 20)\n"
        "  --stream-enhanced       While recording, stream enhanced output instead of raw grayscale\n"
        "  --port <P>              MJPEG server port (default: 8080)\n"
        "  --cam <index>           Camera index (default: 0)\n";
}

static Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--record" && i + 1 < argc) {
            opt.recordMode = true;
            opt.recordPath = argv[++i];
        } else if (a == "--seconds" && i + 1 < argc) {
            opt.recordSeconds = std::max(1, std::stoi(argv[++i]));
        } else if (a == "--stream-enhanced") {
            opt.streamEnhancedWhileRecord = true;
        } else if (a == "--port" && i + 1 < argc) {
            opt.port = std::stoi(argv[++i]);
        } else if (a == "--cam" && i + 1 < argc) {
            opt.camIndex = std::stoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            printUsage(argv[0]);
            std::exit(1);
        }
    }
    if (opt.recordMode && opt.recordPath.empty()) {
        std::cerr << "Error: --record requires a path\n";
        printUsage(argv[0]);
        std::exit(1);
    }
    return opt;
}

// ================== NIGHT VISION LOGIC CLASS ==================
class NightVisionCam {
private:
    cv::VideoCapture cap;
    cv::Ptr<cv::CLAHE> clahe;
    cv::Mat sharpen_kernel;

    // CLAHE settings (your defaults)
    const double CLAHE_CLIP_LIMIT = 4.0;
    const cv::Size CLAHE_TILE_GRID = cv::Size(8, 8);

    // Options
    Options opt;

    // Recorder
    cv::VideoWriter writer;
    bool writerReady = false;
    std::chrono::steady_clock::time_point recordStart;

public:
    explicit NightVisionCam(const Options& options) : opt(options) {
        cap.open(opt.camIndex, cv::CAP_V4L2);

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera.\n";
            std::exit(-1);
        }

        clahe = cv::createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID);

        sharpen_kernel = (cv::Mat_<float>(3, 3) <<
                           0, -1,  0,
                          -1,  5, -1,
                           0, -1,  0);

        if (opt.recordMode) {
            // MJPG AVI is simple + portable for OpenCV
            int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
            double fps = 30.0; // keep consistent with your stream pacing
            writer.open(opt.recordPath, fourcc, fps, cv::Size(640, 480), true);
            if (!writer.isOpened()) {
                std::cerr << "Error: Could not open VideoWriter at " << opt.recordPath << "\n";
                std::exit(-1);
            }
            writerReady = true;
            recordStart = std::chrono::steady_clock::now();
            std::cout << "🎥 Recording RAW dataset to: " << opt.recordPath
                      << " (" << opt.recordSeconds << "s)\n";
            if (opt.streamEnhancedWhileRecord) {
                std::cout << "📡 Streaming: ENHANCED preview while recording\n";
            } else {
                std::cout << "📡 Streaming: RAW GRAYSCALE preview while recording\n";
            }
        }
    }

    void run() {
        cv::Mat frame, gray, equalized, denoised, sharp, boosted;

        std::cout << "✅ Night Vision Started. Mode: "
                  << (opt.recordMode ? "RECORD" : "NORMAL") << "\n";

        while (isRunning) {
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "[CAM] Empty frame, retrying...\n";
                usleep(10000);
                continue;
            }

            // 1) Rotate (keep same as your pipeline)
            cv::rotate(frame, frame, cv::ROTATE_180);

            // If recording: write RAW rotated BGR frame (no enhancement)
            if (opt.recordMode && writerReady) {
                writer.write(frame);

                // Stop after N seconds
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - recordStart).count();
                if (elapsed >= opt.recordSeconds) {
                    std::cout << "⏹ Finished recording (" << opt.recordSeconds << "s). Stopping.\n";
                    isRunning = false; // will stop server too
                }
            }

            // For streaming preview, choose raw gray or enhanced while recording
            if (opt.recordMode && !opt.streamEnhancedWhileRecord) {
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    gray.copyTo(globalFrame);
                }
                // keep loop timing ~30 FPS
                usleep(1000); // small sleep; MJPEG sender controls ~30fps anyway
                continue;
            }

            // ---- Your enhancement pipeline ----
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            clahe->apply(gray, equalized);
            cv::GaussianBlur(equalized, denoised, cv::Size(3, 3), 0);
            cv::filter2D(denoised, sharp, -1, sharpen_kernel);
            cv::convertScaleAbs(sharp, boosted, 1.25, 0);

            // Update stream
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                boosted.copyTo(globalFrame);
            }

            // optional small sleep to reduce CPU usage
            usleep(1000);
        }
    }
};

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);

    // Start MJPEG server in background
    std::thread serverThread(mjpegServer, opt.port);

    // Start camera + processing (blocking)
    NightVisionCam nv(opt);
    nv.run();

    // Cleanup
    isRunning = false;
    if (serverThread.joinable())
        serverThread.join();

    return 0;
}
