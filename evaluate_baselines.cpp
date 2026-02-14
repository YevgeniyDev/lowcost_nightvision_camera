#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

enum Method {
    RAW_GRAY = 0,
    CLAHE_ONLY,
    BILATERAL_CLAHE,
    NLM_CLAHE,
    RETINEX_SSR,
    PROPOSED
};

static std::string methodName(Method m) {
    switch (m) {
        case RAW_GRAY: return "RawGray";
        case CLAHE_ONLY: return "CLAHE";
        case BILATERAL_CLAHE: return "Bilateral+CLAHE";
        case NLM_CLAHE: return "NLM+CLAHE";
        case RETINEX_SSR: return "RetinexSSR";
        case PROPOSED: return "Proposed";
        default: return "Unknown";
    }
}

// ----- Metrics (no-reference) -----
static double entropy8u(const cv::Mat& img8u) {
    CV_Assert(img8u.type() == CV_8U);
    int histSize = 256;
    float range[] = {0.f, 256.f};
    const float* histRange = { range };
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

// ----- Simple Retinex SSR -----
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

// ----- Apply pipeline method (input: BGR 640x480 or Gray) -----
static cv::Mat applyMethod(const cv::Mat& frameBGR,
                           Method m,
                           const cv::Ptr<cv::CLAHE>& clahe,
                           const cv::Mat& sharpen_kernel)
{
    // Input is recorded BGR already rotated in your recording mode (recommended).
    // If you recorded unrotated frames, rotate here to match your main pipeline.
    cv::Mat gray;
    if (frameBGR.channels() == 3) cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);
    else gray = frameBGR;

    if (m == RAW_GRAY) return gray;

    if (m == CLAHE_ONLY) {
        cv::Mat eq;
        clahe->apply(gray, eq);
        return eq;
    }

    if (m == BILATERAL_CLAHE) {
        cv::Mat den, eq;
        // parameters are "reasonable" for 640x480 IR/noisy scenes
        cv::bilateralFilter(gray, den, /*d=*/7, /*sigmaColor=*/50, /*sigmaSpace=*/50);
        clahe->apply(den, eq);
        return eq;
    }

    if (m == NLM_CLAHE) {
        cv::Mat den, eq;
        // h controls strength; tune if needed but keep fixed for fairness
        cv::fastNlMeansDenoising(gray, den, /*h=*/10, /*templateWindow=*/7, /*searchWindow=*/21);
        clahe->apply(den, eq);
        return eq;
    }

    if (m == RETINEX_SSR) {
        // You can optionally CLAHE after SSR, but keep it simple for baseline
        return retinexSSR(gray, 30.0);
    }

    // PROPOSED (your pipeline):
    cv::Mat eq, den, sharp, boosted;
    clahe->apply(gray, eq);
    cv::GaussianBlur(eq, den, cv::Size(3,3), 0);
    cv::filter2D(den, sharp, -1, sharpen_kernel);
    cv::convertScaleAbs(sharp, boosted, 1.25, 0);
    return boosted;
}

struct Accum {
    double ms = 0.0;
    double ent = 0.0;
    double edge = 0.0;
    double lapv = 0.0;
    double contr = 0.0;
    long long n = 0;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./evaluate_baselines <input_video.avi> [out.csv] [save_examples(0/1)]\n";
        return 1;
    }

    std::string inPath = argv[1];
    std::string outCsv = (argc >= 3) ? argv[2] : "results.csv";
    bool saveExamples = (argc >= 4) ? (std::stoi(argv[3]) != 0) : true;

    cv::VideoCapture cap(inPath);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << inPath << "\n";
        return 1;
    }

    // Force expected size check (optional)
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Video opened: " << w << "x" << h << "\n";

    // Your CLAHE settings from README/code
    auto clahe = cv::createCLAHE(4.0, cv::Size(8,8));

    // Your sharpening kernel (as in your code)
    cv::Mat sharpen_kernel = (cv::Mat_<float>(3,3) <<
        0, -1,  0,
       -1,  5, -1,
        0, -1,  0);

    std::vector<Method> methods = {
        RAW_GRAY, CLAHE_ONLY, BILATERAL_CLAHE, NLM_CLAHE, RETINEX_SSR, PROPOSED
    };

    std::vector<Accum> acc(methods.size());

    // Grab a few example frames to save visuals (frame indices)
    std::vector<int> exampleIdx = {10, 60, 120}; // adjust if needed
    int frameIdx = 0;

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        for (size_t i = 0; i < methods.size(); i++) {
            cv::TickMeter tm;
            tm.start();
            cv::Mat out = applyMethod(frame, methods[i], clahe, sharpen_kernel);
            tm.stop();

            // Ensure 8U grayscale
            if (out.type() != CV_8U) out.convertTo(out, CV_8U);

            acc[i].ms += tm.getTimeMilli();
            acc[i].ent += entropy8u(out);
            acc[i].edge += edgeStrengthMeanGrad(out);
            acc[i].lapv += laplacianVar(out);
            acc[i].contr += rmsContrast(out);
            acc[i].n++;

            if (saveExamples) {
                for (int ex : exampleIdx) {
                    if (frameIdx == ex) {
                        std::string fn = "example_f" + std::to_string(frameIdx) + "_" + methodName(methods[i]) + ".png";
                        cv::imwrite(fn, out);
                    }
                }
            }
        }

        frameIdx++;
    }

    // Write CSV
    std::ofstream f(outCsv);
    f << "method,avg_ms_per_frame,fps,entropy,edge_strength,laplacian_var,rms_contrast\n";
    for (size_t i = 0; i < methods.size(); i++) {
        if (acc[i].n == 0) continue;
        double n = (double)acc[i].n;
        double avgMs = acc[i].ms / n;
        double fps = (avgMs > 1e-9) ? (1000.0 / avgMs) : 0.0;

        f << methodName(methods[i]) << ","
          << avgMs << ","
          << fps << ","
          << (acc[i].ent / n) << ","
          << (acc[i].edge / n) << ","
          << (acc[i].lapv / n) << ","
          << (acc[i].contr / n)
          << "\n";
    }
    f.close();

    std::cout << "Saved CSV: " << outCsv << "\n";
    if (saveExamples) std::cout << "Saved example images: example_f*_*.png\n";
    return 0;
}
