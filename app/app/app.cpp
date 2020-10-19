#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include<stdint.h>

using namespace cv;

const int fps = 20;

int main(int argc, char** argv)
{
	namedWindow("Okienko Prawie Inteligentne", 1);
	
	Mat frame;
	VideoCapture vid(0);

	if (!vid.isOpened()) {
		return -1;
	}

	while (vid.read(frame)) {
		imshow("Webcam", frame);

		if (waitKey(1000 / fps) == 27)
			break;
	}
	return 0;
}