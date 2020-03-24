#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

cv::Rect getLeftmostEye(std::vector<cv::Rect> &eyes){
    int leftmost = 99999999;
    int leftmostIndex = -1;
    for (int i = 0; i < eyes.size(); i++)
    {
        if (eyes[i].tl().x < leftmost)
        {
            leftmost = eyes[i].tl().x;
            leftmostIndex = i;
        }
    }
    return eyes[leftmostIndex];
}

cv::Vec3f getEyeball(cv::Mat &eye, std::vector<cv::Vec3f> &circles){
    std::vector<int> sums(circles.size(), 0);
    for (int y = 0; y < eye.rows; y++){
        uchar *ptr = eye.ptr<uchar>(y);
        for (int x = 0; x < eye.cols; x++){
            int value = static_cast<int>(*ptr);
            for (int i = 0; i < circles.size(); i++){
                cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
                int radius = (int)std::round(circles[i][2]);
                if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2)){
                    sums[i] += value;
                }
            }
            ++ptr;
        }
    }
    int smallestSum = 9999999;
    int smallestSumIndex = -1;
    for (int i = 0; i < circles.size(); i++)
    {
        if (sums[i] < smallestSum)
        {
            smallestSum = sums[i];
            smallestSumIndex = i;
        }
    }
    return circles[smallestSumIndex];
}

std::vector<cv::Point> centers;

cv::Point stabilize(std::vector<cv::Point> &points, int windowSize){
    float sumX = 0;
    float sumY = 0;
    int count = 0;
    for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
    {
        sumX += points[i].x;
        sumY += points[i].y;
        ++count;
    }
    if (count > 0)
    {
        sumX /= count;
        sumY /= count;
    }
    return cv::Point(sumX, sumY);
}

void detectEyes(cv::Mat &frame, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade){
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, CV_BGR2GRAY); // convert image to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(150, 150));

    if (faces.size() == 0) return; // none face was detected
    cv::Mat face = frame(faces[0]); // crop the face
    std::vector<cv::Rect> eyes;
    eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(150, 150)); // same thing as above

    // visualize face and eyes
    rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
    if (eyes.size() != 2) return; // both eyes were not detected
    for (cv::Rect &eye : eyes){
        rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
    }
    cv::Rect eyeRect = getLeftmostEye(eyes);

    // visualize iris
    cv::Mat eye = face(eyeRect); // crop the leftmost eye
    cv::Mat eye_gray;

    // convert it to gray
    cvtColor( eye, eye_gray, CV_BGR2GRAY );

    // histogram equalization
    cv::equalizeHist(eye_gray, eye );

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(eye_gray, circles, CV_HOUGH_GRADIENT, 1, eye_gray.cols / 8, 250, 15, eye_gray.rows / 8, eye_gray.rows / 3);

    if (circles.size() > 0){
        cv::Vec3f eyeball = getEyeball(eye, circles);
        cv::Point center(eyeball[0], eyeball[1]);
        centers.push_back(center);
        center = stabilize(centers, 5); // we are using the last 5

        int radius = (int)std::round(eyeball[2]);

        cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, cv::Scalar(0, 0, 255), 2);
        cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2);
    }
    cv::imshow("Eye", eye);
}

int main()
{
    // Load filters
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade;

    std::string cascadePath = "/usr/local/include/opencv2/data/haarcascades/";

    if (!faceCascade.load( cascadePath + "haarcascade_frontalface_alt.xml" )){
        std::cerr << "Could not load face detector." << std::endl;
        return -1;
    }
    if (!eyeCascade.load( cascadePath + "haarcascade_eye_tree_eyeglasses.xml")){
        std::cerr << "Could not load eye detector." << std::endl;
        return -1;
    }

    // start webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()){
        std::cerr << "Webcam not detected." << std::endl;
        return -1;
    }

    // detection
    cv::Mat frame;
    while (1){
        cap >> frame; // outputs the webcam image to a Mat
        cv::imshow("Webcam", frame); // displays the Mat
        if (cv::waitKey(30) >= 0) break; // takes 30 frames per second. if the user presses any button, it stops from showing the webcam

        detectEyes(frame, faceCascade, eyeCascade);
        cv::imshow("Webcam", frame);
        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}