// java
package org.firstinspires.ftc.teamcode.kronbot.utils.detection.test;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SquareEndGame extends OpenCvPipeline {

    private final Mat hsvMat = new Mat();
    private final Mat colorMaskMat = new Mat();
    private final Mat contourHierarchyMat = new Mat();
    private final List<MatOfPoint> contourList = new ArrayList<>();

    // minimum area to consider (adjust as needed)
    private static final double MIN_QUADRILATERAL_AREA = 1000.0;

    // HSV ranges tuned for very bright lighting to detect blue
    // H: 100-130, S: 180-255, V: 200-255
    private static final Scalar HSV_LOWER_BLUE = new Scalar(80, 20, 100);
    private static final Scalar HSV_UPPER_BLUE = new Scalar(135, 255, 255);

    // Expose a simple count of detected contours (updated each frame)
    private volatile int detectedQuadCount = 0;

    @Override
    public Mat processFrame(Mat frame) {
        Imgproc.cvtColor(frame, hsvMat, Imgproc.COLOR_RGB2HSV);
        // Use named constants for the inRange bounds (tuned for bright blue)
        Core.inRange(hsvMat, HSV_LOWER_BLUE, HSV_UPPER_BLUE, colorMaskMat);

        Mat morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(colorMaskMat, colorMaskMat, Imgproc.MORPH_OPEN, morphKernel);
        Imgproc.morphologyEx(colorMaskMat, colorMaskMat, Imgproc.MORPH_CLOSE, morphKernel);

        contourList.clear();
        Imgproc.findContours(colorMaskMat, contourList, contourHierarchyMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        MatOfPoint largestQuadrilateral = null;
        double largestArea = 0.0;

        for (MatOfPoint contour : contourList) {
            MatOfPoint2f contourFloat = new MatOfPoint2f(contour.toArray());
            double peri = Imgproc.arcLength(contourFloat, true);
            MatOfPoint2f approxCurve2f = new MatOfPoint2f();
            Imgproc.approxPolyDP(contourFloat, approxCurve2f, 0.02 * peri, true);

            int vertices = (int) approxCurve2f.total();
            // corrected: look for 4 vertices (quadrilateral)
            if (vertices == 4) {
                MatOfPoint approxPolygon = new MatOfPoint(approxCurve2f.toArray());
                if (Imgproc.isContourConvex(approxPolygon)) {
                    double area = Math.abs(Imgproc.contourArea(approxPolygon));
                    if (area > MIN_QUADRILATERAL_AREA && area > largestArea) {
                        largestArea = area;
                        if (largestQuadrilateral != null) {
                            largestQuadrilateral.release();
                        }
                        largestQuadrilateral = approxPolygon; // take ownership; do not release here
                    } else {
                        approxPolygon.release();
                    }
                } else {
                    approxPolygon.release();
                }
            }

            contourFloat.release();
            approxCurve2f.release();
        }

        if (largestQuadrilateral != null) {
            Imgproc.drawContours(frame, Arrays.asList(largestQuadrilateral), -1, new Scalar(0, 255, 0), 3);
            detectedQuadCount = 1;
            largestQuadrilateral.release();
        } else {
            detectedQuadCount = 0;
        }

        morphKernel.release();
        // return the original frame (with contours drawn) so the overlays are visible
        return frame;
    }

    /**
     * Returns the number of contours considered as squares/rectangles detected in the last processed frame.
     */
    public int getDetectedQuadCount() {
        return detectedQuadCount;
    }

}
