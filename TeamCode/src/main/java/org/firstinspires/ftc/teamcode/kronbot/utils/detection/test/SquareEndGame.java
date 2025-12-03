package org.firstinspires.ftc.teamcode.kronbot.utils.detection.test;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;
import java.util.ArrayList;
import java.util.List;

public class SquareEndGame extends OpenCvPipeline {

    private final Mat hsv = new Mat();
    private final Mat mask = new Mat();
    private final Mat hierarchy = new Mat();
    private final List<MatOfPoint> contours = new ArrayList<>();

    // Expose a simple count of detected contours (updated each frame)
    private volatile int detectedContourCount = 0;

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        Scalar low  = new Scalar(90, 120, 80);
        Scalar high = new Scalar(125, 255, 255);
        Core.inRange(hsv, low, high, mask);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        contours.clear();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        /*int drawCount = 0;
        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double peri = Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approxCurve, 0.02 * peri, true);

            int vertices = (int) approxCurve.total();
            if (vertices >= 0 && vertices <= 8) {
                Imgproc.drawContours(input, List.of(new MatOfPoint(approxCurve.toArray())), -1, new Scalar(255, 0, 0), 3);
                drawCount++;
            }

        }

        // update volatile count so OpMode can read it
        detectedContourCount = drawCount;*/

        return mask;
    }

    /**
     * Returns the number of contours considered as squares/rectangles detected in the last processed frame.
     */
    public int getDetectedCount() {
        return detectedContourCount;
    }

}