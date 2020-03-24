#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>


int main(int argc, char** argv){
    // Load Pictures
    cv::Mat image;
    cv::Mat image_gray;
    cv::Mat auge;

    image = cv::imread(argv[1], cv::IMREAD_COLOR);


    // Load filters
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    face_cascade.load( "/usr/local/include/opencv2/data/haarcascades/haarcascade_frontalface_alt.xml" );
    eyes_cascade.load( "/usr/local/include/opencv2/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

    // prepare picture for face detection
    std::vector<cv::Rect> faces;
    cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    equalizeHist(image_gray, image_gray);

    // face detection
    face_cascade.detectMultiScale( image_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(80, 80) );
    for( int i = 0; i < faces.size(); i++ ){
        cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( image, center, cv::Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );

        cv::Mat faceROI = image_gray(faces[i]);
        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));
        for(int j=0;j < eyes.size(); j++){
            cv::Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y  +eyes[j].y + eyes[j].height/2 );
            int radius = cvRound((eyes[j].width+eyes[j].height)*0.25);
            circle( image,eye_center,radius,cv::Scalar(255, 0, 0),4,8,0 );
            std::cout << eye_center << std::endl;

            // show eyes in picture separately
            auge = image_gray(cv::Rect(eye_center.x-radius, eye_center.y-radius/2, 2*radius, radius));
            std::string filename = "Auge_" + std::to_string(j);
            cv::imshow( filename, auge );
        }
    }

    cv::imshow( "Detected Face", image );


    // Invert the source image and convert to grayscale
    cv::Mat gray;
    cv::cvtColor(~auge, gray, CV_BGR2GRAY);

    // Convert to binary image by thresholding it
    cv::threshold(gray, gray, 220, 255, cv::THRESH_BINARY);

    // Find all contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Fill holes in each contour
    cv::drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);

    for (int i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        cv::Rect rect = cv::boundingRect(contours[i]);
        int radius = rect.width/2;

        // If contour is big enough and has round shape
        // Then it is the pupil
        if (area >= 30 &&
            std::abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
            std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
        {
            cv::circle(auge, cv::Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255,0,0), 2);
        }
    }

    cv::imshow("imagne", auge);

    cv::waitKey(0);
    return 0;
}