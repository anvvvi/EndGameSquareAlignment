package org.firstinspires.ftc.teamcode.kronbot.manual;

import com.qualcomm.robotcore.eventloop.opmode.OpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import com.acmerobotics.dashboard.FtcDashboard;
import com.acmerobotics.dashboard.telemetry.MultipleTelemetry;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.firstinspires.ftc.teamcode.kronbot.utils.detection.test.SquareEndGame;
import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraFactory;
import org.openftc.easyopencv.OpenCvCameraRotation;

@TeleOp(name = "Test Endgame Detection", group = "Concept")
public class TestEndgameDetectionOp extends OpMode {

    private OpenCvCamera camera;
    private SquareEndGame pipeline;
    private FtcDashboard dashboard;

    @Override
    public void init() {
        // Combine DS telemetry + Dashboard telemetry
        telemetry = new MultipleTelemetry(telemetry, FtcDashboard.getInstance().getTelemetry());
        dashboard = FtcDashboard.getInstance();

        // Create camera (must match Robot Config name)
        int cameraMonitorViewId = hardwareMap.appContext.getResources()
                .getIdentifier("cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());

        camera = OpenCvCameraFactory.getInstance().createWebcam(
                hardwareMap.get(WebcamName.class, "Webcam 1"),
                cameraMonitorViewId);

        pipeline = new SquareEndGame();
        camera.setPipeline(pipeline);

        camera.openCameraDeviceAsync(new OpenCvCamera.AsyncCameraOpenListener() {
            @Override
            public void onOpened() {
                camera.startStreaming(640, 480, OpenCvCameraRotation.UPRIGHT);

                // Stream camera to Dashboard (30 fps)
                if (dashboard != null) {
                    FtcDashboard.getInstance().startCameraStream(camera, 30);
                }
            }

            @Override
            public void onError(int errorCode) {
                telemetry.addData("Camera", "Error opening camera: %d", errorCode);
                telemetry.update();
            }
        });

        telemetry.setMsTransmissionInterval(50);
        telemetry.addData("Status", "Init complete - starting camera");
        telemetry.update();
    }

    @Override
    public void loop() {
        // Read detection result from pipeline
        int detected = 0;
        if (pipeline != null) {
            detected = pipeline.getDetectedCount();
        }

        telemetry.clear();
        telemetry.addData("Detected Squares/Rects", detected);
        telemetry.addData("Instructions", "Press stop to end detection");
        telemetry.update();
    }

    @Override
    public void stop() {
        if (camera != null) {
            try {
                camera.stopStreaming();
            } catch (Exception ignored) {}
        }
        if (dashboard != null) {
            try {
                dashboard.stopCameraStream();
            } catch (Exception ignored) {}
        }
    }
}