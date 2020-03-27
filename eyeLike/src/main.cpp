#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <queue>
#include <stdio.h>
#include <iostream>

#include "constants.h"
#include "findEyeCenter.h"



/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
cv::String face_cascade_name = "../../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::Mat debugImage;

float frameSize;

/**
 * @function main
 */
int main( int argc, const char** argv ) {
    cv::Mat frame;

    // Load the cascades
    if( !face_cascade.load( face_cascade_name) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
    cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
    cv::moveWindow(main_window_name, 400, 100);
    cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
    cv::moveWindow(face_window_name, 10, 100);
    cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
    cv::moveWindow("Right Eye", 10, 600);
    cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
    cv::moveWindow("Left Eye", 10, 800);



    // I make an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
    CvCapture* capture = cvCaptureFromCAM( 0 );
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
#else
    cv::VideoCapture capture(0);
    if( capture.isOpened() ) {
        while( true ) {
            capture.read(frame);
#endif
            // mirror it
            cv::flip(frame, frame, 1);
            frame.copyTo(debugImage);
            frameSize =frame.size().height;

            // Apply the classifier to the frame
            if( !frame.empty() ) {
                detectAndDisplay( frame );
            }
            else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            imshow(main_window_name,debugImage);

            int c = cv::waitKey(10);
            if( (char)c == 'c' ) { break; }
            if( (char)c == 'f' ) {
                imwrite("frame.png",frame);
            }

        }
    }

    return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
    cv::Mat faceROI = frame_gray(face);
    cv::Mat debugFace = faceROI;

    //-- Find eye regions and draw them
    int eye_region_width = face.width * (kEyePercentWidth/100.0);
    int eye_region_height = face.width * (kEyePercentHeight/100.0);
    int eye_region_top = face.height * (kEyePercentTop/100.0);
    cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),eye_region_top,eye_region_width,eye_region_height);
    cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),eye_region_top,eye_region_width,eye_region_height);

    //std::cout << leftEyeRegion.size().width << " x " << leftEyeRegion.size().height << std::endl;

    //-- Find Eye Centers
    cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
    cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
    //std::cout << leftPupil.x << ", " << leftPupil.y << std::endl;

    //scale factor
    float scalefactor = face.height/frameSize;
    float scaleDist = 2.5;

    //left
    int leftEyePos = 0;
    if ( leftEyeRegion.size().width*4/9+(scaleDist*scalefactor) >= leftPupil.x){
        leftEyePos = -1;
    }
    if (leftEyeRegion.size().width*4/9+(scaleDist*scalefactor) < leftPupil.x && leftEyeRegion.size().width*5/9-(scaleDist*scalefactor) > leftPupil.x){
        leftEyePos = 0;
    }
    if (leftEyeRegion.size().width*5/9-(scaleDist*scalefactor) <= leftPupil.x){
        leftEyePos = 1;
    }

    //right
    int rightEyePos = 0;
    if ( leftEyeRegion.size().width*4/9+(scaleDist*scalefactor) >= leftPupil.x){
        leftEyePos = -1;
    }
    if (leftEyeRegion.size().width*4/9+(scaleDist*scalefactor) < leftPupil.x && leftEyeRegion.size().width*5/9-(scaleDist*scalefactor) > leftPupil.x){
        leftEyePos = 0;
    }
    if (leftEyeRegion.size().width*5/9-(scaleDist*scalefactor) <= leftPupil.x){
        leftEyePos = 1;
    }

    if (leftEyePos+rightEyePos<0){
        std::cout << "left" << std::endl;
    }
    if (leftEyePos+rightEyePos==0){
        std::cout << "straight" << std::endl;
    }
    if (leftEyePos+rightEyePos>0){
        std::cout << "right" << std::endl;
    }


    // get corner regions
    cv::Rect leftRightCornerRegion(leftEyeRegion);
    leftRightCornerRegion.width -= leftPupil.x;
    leftRightCornerRegion.x += leftPupil.x;
    leftRightCornerRegion.height /= 2;
    leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
    cv::Rect leftLeftCornerRegion(leftEyeRegion);
    leftLeftCornerRegion.width = leftPupil.x;
    leftLeftCornerRegion.height /= 2;
    leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
    cv::Rect rightLeftCornerRegion(rightEyeRegion);
    rightLeftCornerRegion.width = rightPupil.x;
    rightLeftCornerRegion.height /= 2;
    rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
    cv::Rect rightRightCornerRegion(rightEyeRegion);
    rightRightCornerRegion.width -= rightPupil.x;
    rightRightCornerRegion.x += rightPupil.x;
    rightRightCornerRegion.height /= 2;
    rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
    rectangle(debugFace,leftRightCornerRegion,200);
    rectangle(debugFace,leftLeftCornerRegion,200);
    rectangle(debugFace,rightLeftCornerRegion,200);
    rectangle(debugFace,rightRightCornerRegion,200);
    // change eye centers to face coordinates
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;
    // draw eye centers
    circle(debugFace, rightPupil, 3, 1234);
    circle(debugFace, leftPupil, 3, 1234);

    imshow(face_window_name, faceROI);

}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
    std::vector<cv::Rect> faces;

    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );

    for( int i = 0; i < faces.size(); i++ )
    {
        rectangle(debugImage, faces[i], 1234);
        //std::cout << faces[i].height << std::endl;
    }
    //-- Show what you got
    if (faces.size() > 0) {
        findEyes(frame_gray, faces[0]);
    }

}