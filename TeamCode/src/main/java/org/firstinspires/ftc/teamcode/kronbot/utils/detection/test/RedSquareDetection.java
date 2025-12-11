package org.firstinspires.ftc.teamcode.kronbot.utils.detection.test;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class RedSquareDetection  extends OpenCvPipeline {
    Mat hsvMat = new Mat();
    Mat  redMat = new Mat();

    Scalar lowerRed = new Scalar(140, 120, 40);

    Scalar upperRed = new Scalar(160, 255, 255);

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, hsvMat, Imgproc.COLOR_BGR2HSV);
        Imgproc.medianBlur(hsvMat,hsvMat,9);
        Imgproc.GaussianBlur(hsvMat,hsvMat,new Size(9,9),0);
        Imgproc.morphologyEx(hsvMat,hsvMat, Imgproc.MORPH_OPEN,Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(10,10)));
        Core.inRange(hsvMat, lowerRed,upperRed, redMat);
        return hsvMat;
    }
}
