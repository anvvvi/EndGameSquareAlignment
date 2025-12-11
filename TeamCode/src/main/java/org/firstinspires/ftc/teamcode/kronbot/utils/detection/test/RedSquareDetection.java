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

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, hsvMat, Imgproc.COLOR_BGR2HSV);
        Imgproc.medianBlur(hsvMat,hsvMat,5);
        Core.inRange(hsvMat, new Scalar(140, 120, 40),new Scalar(160, 255, 255), redMat);
        return hsvMat;
    }
}
