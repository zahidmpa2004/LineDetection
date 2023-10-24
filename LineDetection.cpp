#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>

class LaneDetection {
public:
    LaneDetection() {
        lower_black = cv::Scalar(0, 0, 0);
        upper_black = cv::Scalar(227, 100, 70);
        rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        camera = cv::VideoCapture(0); // Use the appropriate camera index
    }

    void run() {
        std::cout << "Started" << std::endl;
        cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("New Image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Line Image", cv::WINDOW_AUTOSIZE);

        while (true) {
            cv::Mat frame;
            camera >> frame;

            if (frame.empty()) {
                break;
            }

            cv::Size kernelSize(3, 3);
            cv::GaussianBlur(frame, frame, kernelSize, 0);

            cv::Mat hsvImage;
            cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);
            cv::Mat thresholded;
            cv::inRange(hsvImage, lower_black, upper_black, thresholded);

            cv::dilate(thresholded, thresholded, rectKernel, cv::Point(-1, -1), 1);

            int lowThreshold = 200;
            int highThreshold = 400;
            cv::Mat cannyEdges;
            cv::Canny(frame, cannyEdges, lowThreshold, highThreshold);

            cv::Mat roiImage = regionOfInterest(cannyEdges);
            std::vector<cv::Vec4i> lineSegments = detectLineSegments(roiImage);
            std::vector<std::vector<int>> laneLines = averageSlopeIntercept(frame, lineSegments);
            cv::Mat lineImage = displayLines(frame, laneLines);

            cv::imshow("Frame", frame);
            cv::imshow("New Image", roiImage);
            cv::imshow("Line Image", lineImage);

            int key = cv::waitKey(30);
            if (key == 'q' || key == 27) {
                break;
            }
        }

        cv::destroyAllWindows();
        camera.release();
        std::cout << "Stopped" << std::endl;
    }

private:
    cv::VideoCapture camera;
    cv::Scalar lower_black;
    cv::Scalar upper_black;
    cv::Mat rectKernel;

    cv::Mat regionOfInterest(const cv::Mat& edges) {
        int height = edges.rows;
        int width = edges.cols;
        cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8U);

        cv::Point pts[4] = {
            cv::Point(0, height * 1 / 2),
            cv::Point(width, height * 1 / 2),
            cv::Point(width, height),
            cv::Point(0, height)
        };

        cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255));
        cv::Mat croppedEdges;
        cv::bitwise_and(edges, mask, croppedEdges);
        return croppedEdges;
    }

    std::vector<cv::Vec4i> detectLineSegments(const cv::Mat& croppedEdges) {
        double rho = 1;
        double angle = CV_PI / 180;
        int minThreshold = 10;

        std::vector<cv::Vec4i> lineSegments;
        cv::HoughLinesP(croppedEdges, lineSegments, rho, angle, minThreshold, 10, 15);

        return lineSegments;
    }

    std::vector<std::vector<int>> averageSlopeIntercept(const cv::Mat& frame, const std::vector<cv::Vec4i>& lineSegments) {
        std::vector<std::vector<int>> laneLines;
        int height = frame.rows;
        int width = frame.cols;

        std::vector<cv::Vec2d> leftFit;
        std::vector<cv::Vec2d> rightFit;

        double boundary = 1.0 / 3.0;
        double leftRegionBoundary = width * (1.0 - boundary);
        double rightRegionBoundary = width * boundary;

        for (const cv::Vec4i& lineSegment : lineSegments) {
            int x1 = lineSegment[0];
            int y1 = lineSegment[1];
            int x2 = lineSegment[2];
            int y2 = lineSegment[3];

            if (x1 == x2) {
                continue; // Skip vertical line segments
            }

            cv::Vec2d fit = cv::polyfit(cv::Vec2d(x1, x2), cv::Vec2d(y1, y2), 1);
            double slope = fit[0];
            double intercept = fit[1];

            if (slope < 0 && x1 < leftRegionBoundary && x2 < leftRegionBoundary) {
                leftFit.push_back(fit);
            }
            else if (slope > 0 && x1 > rightRegionBoundary && x2 > rightRegionBoundary) {
                rightFit.push_back(fit);
            }
        }

        if (!leftFit.empty()) {
            cv::Vec2d leftFitAverage = cv::mean(leftFit);
            laneLines.push_back(makePoints(frame, leftFitAverage));
        }

        if (!rightFit.empty()) {
            cv::Vec2d rightFitAverage = cv::mean(rightFit);
            laneLines.push_back(makePoints(frame, rightFitAverage));
        }

        return laneLines;
    }

    std::vector<int> makePoints(const cv::Mat& frame, const cv::Vec2d& line) {
        int height = frame.rows;
        int width = frame.cols;
        double slope = line[0];
        double intercept = line[1];
        int y1 = height;
        int y2 = static_cast<int>(y1 * 1.0 / 2.0);
        int x1 = std::max(-width, std::min(2 * width, static_cast<int>((y1 - intercept) / slope)));
        int x2 = std::max(-width, std::min(2 * width, static_cast<int>((y2 - intercept) / slope)));
        return {x1, y1, x2, y2};
    }

    cv::Mat displayLines(const cv::Mat& frame, const std::vector<std::vector<int>>& lines) {
        cv::Mat lineImage = frame.clone();

        for (const std::vector<int>& line : lines) {
            cv::line(lineImage, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 255), 20);
        }

        return lineImage;
    }
};

int main() {
    LaneDetection laneDetection;
    laneDetection.run();
    return 0;
}
