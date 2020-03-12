#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>


int main(int argc, char** argv){

    cv::Mat image;
    cv::Mat image_gray;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);

    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    face_cascade.load( "/usr/local/include/opencv2/data/haarcascades/haarcascade_frontalface_alt.xml" );
    eyes_cascade.load( "/usr/local/include/opencv2/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

// Detect faces
    std::vector<cv::Rect> faces;
    cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    equalizeHist(image_gray, image_gray);
    face_cascade.detectMultiScale( image_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(80, 80) );
    for( int i = 0; i < faces.size(); i++ ){
        cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( image, center, cv::Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );
        // rectangle(image,Point (faces[i].x,faces[i].y),Point (faces[i].x+faces[i].width, faces[i].y+faces[i].height),Scalar(255,0,255),4,8,0);

        cv::Mat faceROI = image_gray(faces[i]);
        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));
        for(int j=0;j < eyes.size(); j++){
            cv::Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y  +eyes[j].y + eyes[j].height/2 );
            int radius = cvRound((eyes[j].width+eyes[j].height)*0.25);
            circle( image,eye_center,radius,cv::Scalar(255, 0, 0),4,8,0 );
        }

    }
    imshow( "Detected Face", image );

    cv::waitKey(0);
    return 0;
}