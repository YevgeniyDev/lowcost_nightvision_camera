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
#include <fstream>
#include <cmath>
#include <filesystem>

// --- GLOBAL SHARED RESOURCES ---
cv::Mat globalFrame;
std::mutex frameMutex;
std::atomic<bool> isRunning(true);

// ================== MJPEG CLIENT HANDLER ==================
void handleClient(int clientSock) {
    char buffer[1024];
    int n = recv(clientSock, buffer, sizeof(buffer)-1, 0);
    if (n <= 0) { close(clientSock); return; }
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
        if (img.empty()) { usleep(10000); continue; }

        std::vector<uchar> bufferJpeg;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 50};
        cv::imencode(".jpg", img, bufferJpeg, params);

        std::string boundary = "--boundarydonotcross\r\n";
        std::string contentType = "Content-Type: image/jpeg\r\n";
        std::string contentLength = "Content-Length: " + std::to_string(bufferJpeg.size()) + "\r\n\r\n";

        if (send(clientSock, boundary.c_str(), boundary.length(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, contentType.c_str(), contentType.length(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, contentLength.c_str(), contentLength.length(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, bufferJpeg.data(), bufferJpeg.size(), MSG_NOSIGNAL) < 0) break;
        if (send(clientSock, "\r\n", 2, MSG_NOSIGNAL) < 0) break;

        usleep(33000); // ~30 fps
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
        perror("Socket failed"); return;
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        perror("setsockopt"); close(server_fd); return;
    }

    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed"); close(server_fd); return;
    }
    if (listen(server_fd, 10) < 0) {
        perror("Listen"); close(server_fd); return;
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

// ================== METHODS ==================
enum Method {
    RAW_GRAY = 0,
    CLAHE_ONLY,
    BILATERAL_CLAHE,
    NLM_CLAHE,
    RETINEX_SSR,        // old min-max SSR
    RETINEX_SSR_PCTL,   // improved SSR baseline
    PROPOSED,           // your original pipeline
    PROPOSED_V2         // new pipeline designed to beat Retinex
};

static std::string methodName(Method m) {
    switch (m) {
        case RAW_GRAY: return "RawGray";
        case CLAHE_ONLY: return "CLAHE";
        case BILATERAL_CLAHE: return "Bilateral+CLAHE";
        case NLM_CLAHE: return "NLM+CLAHE";
        case RETINEX_SSR: return "RetinexSSR";
        case RETINEX_SSR_PCTL: return "RetinexSSR_Pctl";
        case PROPOSED: return "Proposed";
        case PROPOSED_V2: return "ProposedV2";
        default: return "Unknown";
    }
}

static std::string safeFileName(std::string s) {
    for (char &c : s) {
        if (c == '+' || c == ' ') c = '_';
    }
    return s;
}

// ================== METRICS ==================
static double entropy8u(const cv::Mat& img8u) {
    CV_Assert(img8u.type() == CV_8U);
    int histSize = 256;
    float range[] = {0.f, 256.f};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&img8u, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    const double total = (double)img8u.total();
    double ent = 0.0;
    for (int i = 0; i < 256; i++) {
        double p = hist.at<float>(i) / total;
        if (p > 1e-12) ent -= p * std::log2(p);
    }
    return ent;
}

static double edgeStrengthMeanGrad(const cv::Mat& img8u) {
    cv::Mat gx, gy;
    cv::Sobel(img8u, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img8u, gy, CV_32F, 0, 1, 3);
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);
    return cv::mean(mag)[0];
}

static double laplacianVar(const cv::Mat& img8u) {
    cv::Mat lap;
    cv::Laplacian(img8u, lap, CV_32F, 3);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

static double rmsContrast(const cv::Mat& img8u) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(img8u, mean, stddev);
    return stddev[0];
}

static double meanIntensity(const cv::Mat& img8u) {
    return cv::mean(img8u)[0];
}

// ================== RETINEX SSR (YOUR OLD VERSION) ==================
static cv::Mat retinexSSR(const cv::Mat& gray8u, double sigma=30.0) {
    cv::Mat f;
    gray8u.convertTo(f, CV_32F, 1.0/255.0);

    cv::Mat blur;
    cv::GaussianBlur(f, blur, cv::Size(0,0), sigma);

    cv::Mat logI, logB;
    cv::log(f + 1e-6f, logI);
    cv::log(blur + 1e-6f, logB);

    cv::Mat r = logI - logB;

    cv::Mat out;
    cv::normalize(r, out, 0, 255, cv::NORM_MINMAX);
    out.convertTo(out, CV_8U);
    return out;
}

// ================== RETINEX SSR (ROBUST NORMALIZATION) ==================
static cv::Mat retinexSSR_percentile(const cv::Mat& gray8u, double sigma = 30.0,
                                     double pLow = 1.0, double pHigh = 99.0) {
    CV_Assert(gray8u.type() == CV_8UC1);

    cv::Mat f;
    gray8u.convertTo(f, CV_32F, 1.0 / 255.0);

    cv::Mat blur;
    cv::GaussianBlur(f, blur, cv::Size(0,0), sigma);

    const float eps = 1e-6f;
    cv::Mat logI, logB, r;
    cv::log(f + eps, logI);
    cv::log(blur + eps, logB);
    r = logI - logB; // SSR

    // Percentile normalization
    cv::Mat flat = r.reshape(1, 1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);

    const int N = sorted.cols;
    int iLow  = std::max(0, std::min(N - 1, (int)std::floor((pLow  / 100.0) * (N - 1))));
    int iHigh = std::max(0, std::min(N - 1, (int)std::floor((pHigh / 100.0) * (N - 1))));

    float vLow  = sorted.at<float>(0, iLow);
    float vHigh = sorted.at<float>(0, iHigh);
    if (vHigh <= vLow + 1e-12f) return gray8u.clone();

    cv::Mat outF = (r - vLow) / (vHigh - vLow);
    cv::min(outF, 1.0f, outF);
    cv::max(outF, 0.0f, outF);

    cv::Mat out8;
    outF.convertTo(out8, CV_8U, 255.0);
    return out8;
}

// ================== OPTIONS ==================
struct Options {
    int port = 8080;
    int camIndex = 0;
    int secondsPerMethod = 20;
    int warmupSeconds = 2;                // don't log metrics during warmup
    std::string outCsv = "results_live.csv";

    // Visual recording:
    bool recordPerMethod = true;          // RawGray.avi, CLAHE.avi, ...
    bool recordCombined = true;           // combined_run.avi
    std::string videoDir = "videos";      // per-method videos go here
    std::string combinedVideo = "combined_run.avi";

    // Snapshots (for paper figures):
    bool saveSnapshots = true;
    std::string snapshotDir = "snaps";
    std::vector<int> snapshotTimes = {3, 8, 15};

    // Optional: simple pre-normalization to reduce near-black runs (kept OFF by default)
    bool usePreExposureNorm = false;
    double exposureTargetMean = 100.0;     // target mean intensity
    double exposureScaleMin = 0.7;
    double exposureScaleMax = 1.4;
};

static void printUsage(const char* prog) {
    std::cout <<
        "Usage:\n"
        "  " << prog << " [options]\n\n"
        "Options:\n"
        "  --seconds-per-method N     (default 20)\n"
        "  --warmup N                 (default 2) warm-up seconds per method (no logging)\n"
        "  --out results.csv          output CSV filename\n"
        "  --port 8080                MJPEG server port\n"
        "  --cam 0                    camera index\n\n"
        "  --record-per-method 0/1    (default 1)\n"
        "  --record-combined 0/1      (default 1)\n"
        "  --video-dir DIR            (default videos)\n"
        "  --combined-video FILE      (default combined_run.avi)\n\n"
        "  --snapshots 0/1            (default 1)\n"
        "  --snapshot-dir DIR         (default snaps)\n\n"
        "  --preexp 0/1               (default 0) pre-normalize mean intensity\n";
}

static Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--seconds-per-method" && i + 1 < argc) opt.secondsPerMethod = std::max(5, std::stoi(argv[++i]));
        else if (a == "--warmup" && i + 1 < argc) opt.warmupSeconds = std::max(0, std::stoi(argv[++i]));
        else if (a == "--out" && i + 1 < argc) opt.outCsv = argv[++i];
        else if (a == "--port" && i + 1 < argc) opt.port = std::stoi(argv[++i]);
        else if (a == "--cam" && i + 1 < argc) opt.camIndex = std::stoi(argv[++i]);

        else if (a == "--record-per-method" && i + 1 < argc) opt.recordPerMethod = (std::stoi(argv[++i]) != 0);
        else if (a == "--record-combined" && i + 1 < argc) opt.recordCombined = (std::stoi(argv[++i]) != 0);
        else if (a == "--video-dir" && i + 1 < argc) opt.videoDir = argv[++i];
        else if (a == "--combined-video" && i + 1 < argc) opt.combinedVideo = argv[++i];

        else if (a == "--snapshots" && i + 1 < argc) opt.saveSnapshots = (std::stoi(argv[++i]) != 0);
        else if (a == "--snapshot-dir" && i + 1 < argc) opt.snapshotDir = argv[++i];

        else if (a == "--preexp" && i + 1 < argc) opt.usePreExposureNorm = (std::stoi(argv[++i]) != 0);

        else if (a == "--help" || a == "-h") { printUsage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown arg: " << a << "\n"; printUsage(argv[0]); std::exit(1); }
    }
    return opt;
}

// ================== ACCUMULATOR ==================
struct Accum {
    double totalMs = 0.0;
    double ent = 0.0;
    double edge = 0.0;
    double lapv = 0.0;
    double contrast = 0.0;
    double meanI = 0.0;
    long long n = 0;

    void add(double ms, const cv::Mat& out8u) {
        totalMs += ms;
        ent += entropy8u(out8u);
        edge += edgeStrengthMeanGrad(out8u);
        lapv += laplacianVar(out8u);
        contrast += rmsContrast(out8u);
        meanI += meanIntensity(out8u);
        n++;
    }
};

// ================== LABEL HELPER ==================
static void drawLabel(cv::Mat& img8u, const std::string& text) {
    cv::rectangle(img8u, cv::Point(0,0), cv::Point(img8u.cols, 26), cv::Scalar(0), -1);
    cv::putText(img8u, text, cv::Point(8,18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255), 1);
}

// ================== NIGHT VISION COMPARISON ==================
class NightVisionComparison {
private:
    Options opt;
    cv::VideoCapture cap;
    cv::Ptr<cv::CLAHE> claheStrong;
    cv::Ptr<cv::CLAHE> claheMild;
    cv::Mat sharpen_kernel;

    const int W = 640, H = 480;
    const double FPS = 30.0;

    std::vector<Method> methods = {
        RAW_GRAY,
        CLAHE_ONLY,
        BILATERAL_CLAHE,
        NLM_CLAHE,
        RETINEX_SSR,
        RETINEX_SSR_PCTL,
        PROPOSED,
        PROPOSED_V2
    };

    std::vector<Accum> results;

    cv::VideoWriter combinedWriter;
    bool combinedReady = false;

public:
    explicit NightVisionComparison(const Options& o) : opt(o), results(methods.size()) {
        cap.open(opt.camIndex, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera.\n";
            std::exit(-1);
        }

        // Your original CLAHE
        claheStrong = cv::createCLAHE(4.0, cv::Size(8,8));
        // Mild CLAHE (used in ProposedV2)
        claheMild   = cv::createCLAHE(2.0, cv::Size(8,8));

        sharpen_kernel = (cv::Mat_<float>(3, 3) <<
                           0, -1,  0,
                          -1,  5, -1,
                           0, -1,  0);

        if (opt.recordPerMethod) std::filesystem::create_directories(opt.videoDir);
        if (opt.saveSnapshots)   std::filesystem::create_directories(opt.snapshotDir);

        if (opt.recordCombined) {
            int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
            combinedWriter.open(opt.combinedVideo, fourcc, FPS, cv::Size(W,H), false);
            if (!combinedWriter.isOpened()) {
                std::cerr << "Warning: could not open combined video: " << opt.combinedVideo << "\n";
            } else {
                combinedReady = true;
            }
        }
    }

    static cv::Mat denoisedFix(const cv::Mat& in) {
        if (in.type() == CV_8U) return in;
        cv::Mat out; in.convertTo(out, CV_8U);
        return out;
    }

    cv::Mat preExposureNormalize(const cv::Mat& gray8u) {
        if (!opt.usePreExposureNorm) return gray8u;

        double m = meanIntensity(gray8u);
        if (m < 1e-6) return gray8u;

        double scale = opt.exposureTargetMean / m;
        scale = std::max(opt.exposureScaleMin, std::min(opt.exposureScaleMax, scale));

        cv::Mat out;
        gray8u.convertTo(out, CV_8U, scale, 0.0);
        return out;
    }

    // ProposedV2: homomorphic/Retinex-like illumination correction + mild CLAHE + edge-aware sharpening
cv::Mat proposedV2(const cv::Mat& gray8u) {
    CV_Assert(gray8u.type() == CV_8UC1);

    // Optional: pre exposure normalize (helps near-black runs)
    cv::Mat g = preExposureNormalize(gray8u);

    // --- 1) Estimate illumination on a smaller image (fast) ---
    cv::Mat small;
    cv::resize(g, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);  // 1/4 scale

    cv::Mat smallF;
    small.convertTo(smallF, CV_32F, 1.0f);

    // Box filter is faster than large Gaussian; good enough for illumination
    cv::Mat illumSmall;
    // ksize tuned for 1/4 resolution; adjust 21..41 if needed
    cv::boxFilter(smallF, illumSmall, CV_32F, cv::Size(31, 31));

    cv::Mat illum;
    cv::resize(illumSmall, illum, g.size(), 0, 0, cv::INTER_LINEAR);

    // --- 2) Retinex-like ratio correction (much faster than log SSR) ---
    cv::Mat gF;
    g.convertTo(gF, CV_32F, 1.0f);

    cv::Mat ratio = gF / (illum + 1.0f); // +1 avoids div by zero in 8-bit domain

    // --- 3) Fast robust normalization using mean/std (no sort) ---
    // Clamp to [mu - k*std, mu + k*std] then scale to 0..255
    cv::Scalar mu, sd;
    cv::meanStdDev(ratio, mu, sd);

    float m  = (float)mu[0];
    float s  = (float)sd[0];
    float k  = 2.5f;                     // 2.0..3.0 works well
    float lo = m - k * s;
    float hi = m + k * s;
    if (hi <= lo + 1e-6f) return g.clone();

    cv::Mat norm = (ratio - lo) * (255.0f / (hi - lo));
    cv::min(norm, 255.0f, norm);
    cv::max(norm, 0.0f, norm);

    cv::Mat base8;
    norm.convertTo(base8, CV_8U);

    // --- 4) Mild CLAHE (optional, keep mild) ---
    cv::Mat eq;
    claheMild->apply(base8, eq);

    // --- 5) Cheap denoise ---
    cv::Mat den;
    cv::medianBlur(eq, den, 3);

    // --- 6) Detail-gated unsharp (edge-aware, cheap, less noisy) ---
    cv::Mat blur;
    cv::GaussianBlur(den, blur, cv::Size(0,0), 1.0);

    cv::Mat denF2, blurF2;
    den.convertTo(denF2, CV_32F);
    blur.convertTo(blurF2, CV_32F);
    cv::Mat detail = denF2 - blurF2;

    // Gate sharpening by detail magnitude (avoid boosting flat noise)
    cv::Mat absDetail;
    cv::absdiff(denF2, blurF2, absDetail);

    const float t = 6.0f;  // threshold in 8-bit intensity units (tune 4..10)
    cv::Mat mask = (absDetail > t);
    mask.convertTo(mask, CV_32F, 1.0 / 255.0); // 0 or 1

    const float sharpK = 1.0f;  // tune 0.7..1.4
    cv::Mat outF = denF2 + sharpK * detail.mul(mask);

    cv::Mat out8;
    outF.convertTo(out8, CV_8U);
    return out8;
}

    cv::Mat apply(Method m, const cv::Mat& frameBGR) {
        cv::Mat frame = frameBGR.clone();
        cv::rotate(frame, frame, cv::ROTATE_180);

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // (optional) pre-exposure normalization can be applied to all methods for fairness
        // If you want it truly global, uncomment this line:
        // gray = preExposureNormalize(gray);

        if (m == RAW_GRAY) return gray;

        if (m == CLAHE_ONLY) {
            cv::Mat eq;
            claheStrong->apply(gray, eq);
            return eq;
        }

        if (m == BILATERAL_CLAHE) {
            cv::Mat den, eq;
            cv::bilateralFilter(gray, den, 7, 50, 50);
            claheStrong->apply(den, eq);
            return eq;
        }

        if (m == NLM_CLAHE) {
            cv::Mat den, eq;
            cv::fastNlMeansDenoising(gray, den, 10, 7, 21);
            claheStrong->apply(den, eq);
            return eq;
        }

        if (m == RETINEX_SSR) {
            return retinexSSR(gray, 30.0);
        }

        if (m == RETINEX_SSR_PCTL) {
            return retinexSSR_percentile(gray, 30.0, 1.0, 99.0);
        }

        if (m == PROPOSED) {
            // Your original pipeline:
            cv::Mat eq, den, sharp, boosted;
            claheStrong->apply(gray, eq);
            cv::GaussianBlur(eq, den, cv::Size(3, 3), 0);
            cv::filter2D(denoisedFix(den), sharp, -1, sharpen_kernel);
            cv::convertScaleAbs(sharp, boosted, 1.25, 0);
            return boosted;
        }

        if (m == PROPOSED_V2) {
            return proposedV2(gray);
        }

        return gray;
    }

    void run() {
        std::cout << "✅ Real-time comparison + recording started.\n";
        std::cout << "Each method: " << opt.secondsPerMethod << "s (warmup " << opt.warmupSeconds << "s)\n";
        std::cout << "Per-method videos: " << (opt.recordPerMethod ? "ON" : "OFF")
                  << " | Combined video: " << (opt.recordCombined ? "ON" : "OFF")
                  << " | Snapshots: " << (opt.saveSnapshots ? "ON" : "OFF") << "\n";

        for (size_t mi = 0; mi < methods.size() && isRunning; mi++) {
            Method m = methods[mi];
            std::string mName = methodName(m);
            std::cout << "\n=== Method: " << mName << " ===\n";

            cv::VideoWriter methodWriter;
            bool methodReady = false;
            if (opt.recordPerMethod) {
                std::string path = opt.videoDir + "/" + safeFileName(mName) + ".avi";
                int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
                methodWriter.open(path, fourcc, FPS, cv::Size(W,H), false);
                if (!methodWriter.isOpened()) {
                    std::cerr << "Warning: could not open video for " << mName << " at " << path << "\n";
                } else {
                    methodReady = true;
                }
            }

            auto tStart = std::chrono::steady_clock::now();

            while (isRunning) {
                auto now = std::chrono::steady_clock::now();
                int elapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(now - tStart).count();
                if (elapsed >= opt.secondsPerMethod) break;

                cv::Mat frame;
                cap >> frame;
                if (frame.empty()) { usleep(10000); continue; }

                cv::TickMeter tm;
                tm.start();
                cv::Mat out = apply(m, frame);
                tm.stop();

                if (out.empty()) continue;
                if (out.size() != cv::Size(W,H)) cv::resize(out, out, cv::Size(W,H));
                if (out.type() != CV_8U) out.convertTo(out, CV_8U);

                cv::Mat labelled = out.clone();
                drawLabel(labelled, mName + " | t=" + std::to_string(elapsed) + "s");

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    labelled.copyTo(globalFrame);
                }

                if (methodReady)   methodWriter.write(out);
                if (combinedReady) combinedWriter.write(labelled);

                if (opt.saveSnapshots) {
                    for (int st : opt.snapshotTimes) {
                        if (elapsed == st) {
                            std::string fn = opt.snapshotDir + "/snap_" + safeFileName(mName) + "_t" + std::to_string(elapsed) + ".png";
                            cv::imwrite(fn, labelled);
                        }
                    }
                }

                if (elapsed >= opt.warmupSeconds) {
                    results[mi].add(tm.getTimeMilli(), out);
                }
            }

            if (results[mi].n > 0) {
                double n = (double)results[mi].n;
                double avgMs = results[mi].totalMs / n;
                double fps = 1000.0 / avgMs;
                double meanI = results[mi].meanI / n;

                std::cout << "Frames logged: " << results[mi].n
                          << " | avg " << avgMs << " ms/frame"
                          << " | ~" << fps << " FPS"
                          << " | mean intensity " << meanI << "\n";

                if (meanI < 3.0) {
                    std::cout << "⚠ Warning: mean intensity is very low (near-black frames). "
                              << "This run may be unreliable (auto-exposure/IR issue).\n";
                }
            } else {
                std::cout << "⚠ No frames logged for this method.\n";
            }
        }

        writeCsv();
        std::cout << "✅ Done. CSV saved to " << opt.outCsv << "\n";
        isRunning = false;
    }

    void writeCsv() {
        std::ofstream f(opt.outCsv);
        f << "method,avg_ms_per_frame,fps,entropy,edge_strength,laplacian_var,rms_contrast,mean_intensity,frames\n";

        for (size_t i = 0; i < methods.size(); i++) {
            if (results[i].n == 0) continue;
            double n = (double)results[i].n;

            double avgMs = results[i].totalMs / n;
            double fps = 1000.0 / avgMs;

            f << methodName(methods[i]) << ","
              << avgMs << ","
              << fps << ","
              << (results[i].ent / n) << ","
              << (results[i].edge / n) << ","
              << (results[i].lapv / n) << ","
              << (results[i].contrast / n) << ","
              << (results[i].meanI / n) << ","
              << results[i].n << "\n";
        }
        f.close();
    }
};

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);

    std::thread serverThread(mjpegServer, opt.port);

    NightVisionComparison cmp(opt);
    cmp.run();

    isRunning = false;
    if (serverThread.joinable())
        serverThread.join();

    return 0;
}
